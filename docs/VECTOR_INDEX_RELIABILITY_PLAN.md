# Plan: đồng bộ vector tự phục hồi cho AGY Memory Engine

Ngày lập: 2026-09-19. Trạng thái: đề xuất để giao cho agent triển khai.

Tài liệu này chưa triển khai tính năng, chưa cập nhật checkout, chưa chạy migration hoặc reindex database thật. Yêu cầu của Kai ở bước này là chỉ viết plan. Agent nhận tài liệu cần có yêu cầu triển khai riêng trước khi thực hiện các bước thay đổi hệ thống.

## 1. Kết quả cần đạt

Khi nội dung memory thay đổi, vector của nội dung cũ phải ngừng tham gia tìm kiếm ngữ nghĩa. Engine tự tạo vector mới, thử lại khi lỗi và chỉ nhận kết quả còn khớp với nội dung hiện tại. Trong lúc chờ, memory vẫn được lưu đầy đủ và có thể tìm bằng FTS.

Áp dụng cho cả `memories`, `episodes`, `learnings`; cho các đường đọc MCP và dashboard. Giữ CLI `prefetch` dùng FTS như thiết kế hiện tại.

Không thay nội dung memory, taxonomy, graph semantics, ranking RRF hoặc extraction policy trong tính năng này. Không cài thêm scheduler, daemon hay dịch vụ bên ngoài khi chưa xác định cách tích hợp với lịch chạy hiện có.

## 2. Bằng chứng và baseline

Repository: `/Users/__blitzzz/Documents/GitHub/agy-memory-engine`.

Code đã khảo sát là commit `c69a261becf1241d86fd7e68197b3ebfc88a426c`, qua Git object `upstream/main`. Ngày lập plan, checkout vẫn ở `94c1cc7014a55bb671f834b921c3a464fa54b6f5`, chậm remote-tracking ref đó 31 commit. Không fetch/pull trong bước lập plan. Agent triển khai phải kiểm tra lại upstream và bắt đầu trên bản đã merge, không xây tính năng trên checkout cũ.

Quan sát database ngày 2026-09-18, không phải cam kết về số lượng hiện tại:

| Loại | Nội dung | Vector | Thiếu vector |
| --- | ---: | ---: | ---: |
| Facts | 156 | 151 | 5 |
| Episodes | 90 | 87 | 3 |
| Learnings | 114 | 106 | 8 |
| Tổng | 360 | 344 | 16 |

Tái hiện độc lập bằng SQLite trong RAM và các hàm lấy từ commit trên: khi ép `embed_text()` trả `None`, `upsert_fact()` vẫn lưu nội dung mới; bản ghi cũ giữ vector cũ; bản ghi mới không có vector. Chưa xác định nguyên nhân lịch sử của toàn bộ 16 bản ghi thiếu vector.

Các điểm nối đã xác minh:

| File hoặc hàm | Hiện trạng liên quan |
| --- | --- |
| `agy_memory.py`: `_writer`, `upsert_fact`, `upsert_episode`, `upsert_learning` | Tính vector ngay trong đường ghi nội dung; kết quả thất bại của `upsert_vector` không được xử lý thành công việc chờ retry. |
| `embedder.py`: `build_text_repr`, `embed_text`, `upsert_vector` | Có chuẩn hóa text dùng chung; lỗi embedding có thể trả `None`; vector cũ chưa bị vô hiệu hóa. |
| `schema.py` | Đã có `entity_revisions`, tombstone qua revision, `meta_generation`, khóa maintenance và schema version `211`. |
| `agy_memory_mcp.py`: `search_memory` | Có FTS, vector search, RRF; chưa kiểm tra vector khớp phiên bản nguồn. |
| `dashboard.py`: `_handle_search` | Có đường vector search riêng, cũng phải áp dụng cùng hợp đồng về độ mới. |
| `scripts/reindex_vectors.py` | Reindex trực tiếp, chưa dùng job có lease và revision fencing. |
| `scripts/queue_cli.py`, `memory_worker.py` | Ghi memory thông qua các hàm upsert; không được làm hỏng atomic commit và receipt của extraction. |
| `docs/scheduled_task_instruction.md` | Workflow hiện có dùng scheduled invocation; có nhánh thoát sớm khi turn queue rỗng. |

Nguồn baseline: [đường ghi](https://github.com/HydStAn/agy-memory-engine/blob/c69a261becf1241d86fd7e68197b3ebfc88a426c/agy_memory.py#L609), [vector embedding](https://github.com/HydStAn/agy-memory-engine/blob/c69a261becf1241d86fd7e68197b3ebfc88a426c/embedder.py#L120), [schema](https://github.com/HydStAn/agy-memory-engine/blob/c69a261becf1241d86fd7e68197b3ebfc88a426c/schema.py#L281).

## 3. Các nguyên tắc bắt buộc

1. Nội dung và yêu cầu index được commit trong cùng transaction của `memory.db`. Không dùng `turn_queue.db` làm outbox vector, vì sẽ phát sinh bài toán atomicity giữa hai database.
2. Sau commit sửa memory, một truy vấn bắt đầu trên snapshot mới không được dùng vector phiên bản cũ. Truy vấn đang đọc snapshot cũ có thể hoàn tất với cặp nội dung và vector cũ còn nhất quán.
3. Không tính embedding, tải model hoặc gọi mạng khi đang giữ write transaction. Claim và publish là hai transaction ngắn; embedding nằm giữa chúng.
4. Chỉ publish nếu entity còn tồn tại và revision, database generation, model fingerprint, lease token và thời hạn lease đều hợp lệ tại thời điểm ghi dưới khóa.
5. Vector thiếu metadata không được mặc định coi là hợp lệ. Không dùng tổng số vector bằng tổng số memory làm bằng chứng đầy đủ về độ mới.
6. Ghi vector, metadata xác nhận phiên bản và hoàn tất job phải atomic. Worker crash không được tạo trạng thái báo ready nhưng chưa có vector, hoặc đã mất job mà chưa publish.
7. FTS tiếp tục hoạt động khi embedding bị lỗi hoặc vector search bị tắt. Outbox không được mất; trạng thái suy giảm phải quan sát được.
8. Đọc nguồn, kiểm tra metadata và lấy kết quả tìm kiếm phải dùng cùng database và snapshot nhất quán. Giữ cách ly giữa các database/profile của dashboard.

## 4. Thiết kế dữ liệu

Tận dụng `entity_revisions` và `meta_generation` hiện có. Không tạo một bộ đếm revision cạnh tranh với bộ đếm đang dùng cho extraction. Các tên bảng mới bên dưới là đề xuất; agent có thể đổi tên nhưng phải giữ các ràng buộc.

| Thành phần | Dữ liệu cần có |
| --- | --- |
| `vector_index_config` | Cấu hình index đang active trong database: model định danh cụ thể, dimension, phiên bản `build_text_repr`, fingerprint của cấu hình. |
| `vector_index_state` | Khóa `(entity_type, entity_id)`; `indexed_revision`, database generation, model fingerprint, hash của text thực sự đã embed, `indexed_at`. |
| `vector_index_jobs` | Khóa duy nhất `(entity_type, entity_id)`; thao tác index hoặc delete; `desired_revision`; trạng thái pending/claimed/retry/blocked; attempts; `next_attempt_at`; `lease_token`; `lease_expires_at`; lỗi gần nhất và các mốc thời gian. |

Mỗi entity có tối đa một công việc hiện hành. Sửa liên tiếp cập nhật đích đến phiên bản mới nhất, không tạo hàng trăm job cũ. Worker đang xử lý bản cũ bị mất quyền publish, nhưng không được xóa hoặc hoàn tất job của bản mới.

`text_hash` dùng biểu diễn chuẩn từ `build_text_repr`, không hash riêng trường nội dung rồi bỏ qua category, keywords hoặc stance. Model fingerprint phải phân biệt model, artifact/revision nếu có, dimension và phiên bản biểu diễn text. Không tái sử dụng vector chỉ vì cùng ID hoặc cùng số chiều.

Database generation dùng để chặn kết quả từ trước khi restore. Snapshot được restore sẽ được coi là cần đối soát và tạo lại metadata/vector theo generation hiện tại. Không tự gắn generation mới cho vector cũ khi chưa xác minh.

## 5. Đường ghi và xử lý vector cũ

Trong transaction ghi nội dung:

1. Ghi memory và tăng revision.
2. Upsert job cho revision mới nhất.
3. Metadata của vector cũ không còn khớp revision, nên vector đó mất tư cách tham gia tìm kiếm ngay sau commit.

Ưu tiên thực hiện phần revision và outbox trong trigger SQLite thuần SQL dùng chung cho các bảng nguồn. Cần tích hợp với trigger revision hiện có; không dựa vào thứ tự chạy của nhiều trigger độc lập. Không gọi Python, hash UDF, model hay extension vector từ trigger enqueue.

Rà tất cả writer, gồm upsert, extraction, consolidation, migration, đổi ID và đường SQL trực tiếp trong repository. Nếu agent chọn application-level enqueue, phải chứng minh không còn writer nào có thể bỏ qua; periodic repair không thay thế được atomic enqueue.

Vector cũ có thể tạm còn vật lý trong `vec_*`, nhưng bị loại bởi điều kiện freshness của truy vấn. Sau khi vector mới được chấp nhận, worker xóa vector cũ, chèn vector mới, cập nhật metadata và hoàn tất job trong một transaction. Nếu bất kỳ bước nào thất bại, rollback toàn bộ.

Xóa entity tạo công việc dọn vector và metadata. Entity đã xóa không được xuất hiện trong truy vấn dù cleanup chưa chạy. Giữ revision tombstone để worker cũ không hồi sinh vector sau khi ID được xóa rồi tạo lại. Đổi ID phải xử lý cả ID cũ và ID mới.

Các trigger cascade `trg_vec_*` hiện có cũng cần được rà khi hỗ trợ chế độ thiếu extension. Đường ghi FTS không được lỗi chỉ vì trigger cố thao tác `vec0` trong connection chưa load `sqlite-vec`.

## 6. Worker và cơ chế retry

Thêm module xử lý dùng chung, ví dụ `vector_index.py`, cùng CLI hữu hạn, ví dụ `scripts/vector_index_cli.py`. Đây là tên dự kiến, chưa phải file hoặc command đang tồn tại.

Một lần xử lý:

1. Trong transaction ngắn, claim job đến hạn bằng token duy nhất, lease có hạn và giới hạn batch. Chụp nội dung, revision, generation và model fingerprint nhất quán.
2. Đóng transaction và connection không cần thiết, rồi tính embedding ngoài write lock. Cache phải phân biệt model fingerprint và không giữ lỗi `None` vĩnh viễn như một kết quả thành công.
3. Mở transaction ngắn mới. Đọc lại trạng thái hiện tại và kiểm tra tất cả điều kiện ở mục 3. Không chỉ kiểm tra lease trước khi chạy model.
4. Nếu khớp, publish vector và hoàn tất job atomically. Nếu source đã đổi hoặc lease mất hiệu lực, bỏ kết quả; để job mới tiếp tục xử lý.
5. Lỗi tạm thời dùng exponential backoff có giới hạn và jitter. Lỗi cấu hình không retry nóng; chuyển blocked kèm lý do, có đường phục hồi khi cấu hình được sửa.

Lease hết hạn cho phép worker khác claim lại. Worker chết sau claim hoặc trong lúc embed không làm mất công việc. Xử lý lặp phải idempotent. Giới hạn attempts trong một invocation để tránh một bản ghi lỗi chiếm toàn bộ worker.

CLI cần có các khả năng sau, tên flag có thể điều chỉnh:

- `status`: đọc số lượng và tình trạng, không tự migration hoặc repair.
- `run`: drain có giới hạn batch/thời gian, exit rõ ràng, không chạy daemon.
- `reconcile --dry-run`: chỉ báo thiếu, stale, orphan, metadata lệch và việc cần enqueue.
- `reconcile --apply`: enqueue có kiểm soát, không gọi embedding trong transaction quét.

`memory_worker.py` và scheduled workflow dùng chung implementation. Vector queue phải được xử lý cả khi turn queue rỗng hoặc đang debounce. Điều chỉnh nhánh thoát sớm trong scheduled instruction để không bỏ đói vector jobs. Không khởi động thêm LLM agent chỉ để tính local embedding; tích hợp vào lịch chạy hiện có sau khi kiểm tra cách vận hành thực tế.

## 7. Đường tìm kiếm

Chia sẻ hàm đọc vector hợp lệ giữa MCP và dashboard. Một vector đủ điều kiện khi source tồn tại và metadata khớp source revision, generation cùng model fingerprint active. Thiếu metadata hoặc cấu hình không tương thích thì bỏ semantic candidate đó, vẫn giữ FTS.

Phải lọc tập ID hợp lệ trước khi chọn top-k trong `sqlite-vec`. Không chỉ JOIN/filter sau khi `k` đã được áp dụng, vì vector stale có thể chiếm hết top-k rồi bị loại, làm mất kết quả hợp lệ ở phía sau.

Kiểm chứng trong RAM ngày lập plan, dùng `sqlite-vec 0.1.9`, ba vector và `k=2`:

- Vector gần nhất là stale; hai vector tiếp theo hợp lệ.
- JOIN với bảng ID hợp lệ sau KNN chỉ trả một kết quả.
- Điều kiện `id IN (SELECT id FROM eligible)` bên trong truy vấn KNN trả đủ hai kết quả hợp lệ.

Agent phải kiểm chứng lại với phiên bản dependency tối thiểu được hỗ trợ, hiện requirements baseline là `sqlite-vec>=0.1.6`. Nếu cần tăng minimum version, ghi rõ lý do và kiểm thử. Không giả định một JOIN bất kỳ được SQLite đẩy xuống trước KNN. Khi không thể bảo đảm lọc đúng, fallback FTS và báo degraded thay vì dùng vector stale.

Ví dụ dạng truy vấn cần kiểm chứng tiếp trên schema thật:

```sql
SELECT id, distance
FROM vec_memories
WHERE embedding MATCH :query_embedding
  AND k = :limit
  AND id IN (SELECT entity_id FROM eligible_fact_ids)
ORDER BY distance;
```

`eligible_fact_ids` ở ví dụ là tập ID được suy ra từ source và metadata, không phải bảng đã tồn tại. Kiểm tra đủ ba loại entity, limit, ít candidate hợp lệ, không có candidate và stale candidate áp đảo.

Tính query embedding trước read transaction dài; sau đó xác nhận fingerprint còn khớp với cấu hình trong snapshot đọc. Nếu cấu hình đã đổi, tính lại ngoài transaction hoặc fallback FTS. Source hydration và RRF không được ghép metadata từ snapshot này với nội dung từ snapshot khác.

## 8. Migration, backfill và vận hành

Migration nâng schema từ `211` lên version tiếp theo còn trống ở baseline lúc triển khai. Migration phải idempotent và không gọi model. Tạo bảng/trigger, seed jobs cho dữ liệu cũ theo batch; không gán metadata ready cho vector chưa xác minh.

Vector legacy chưa có version được coi là unverified. Trong thời gian backfill, những bản ghi này dùng FTS. Worker dựng lại từ nội dung hiện tại và lần lượt đưa từng vector hợp lệ vào phục vụ. Chấp nhận suy giảm semantic coverage tạm thời, có số liệu tiến độ; không để index trông như đã sẵn sàng khi chưa xong.

Đưa `scripts/reindex_vectors.py` về cùng cơ chế enqueue, fencing và publish, tránh giữ một lối ghi vector không kiểm tra revision. Lệnh phải phân biệt đã enqueue với đã index xong, và trả lỗi/degraded thật khi model hoặc index lỗi.

Worker đối soát định kỳ cần phát hiện:

- Source thiếu vector hoặc thiếu metadata.
- Metadata có revision/model/generation cũ.
- Vector hoặc metadata không còn source.
- Job mất lease, job bị bỏ sót và trạng thái ready thiếu vector vật lý.

Khi thay model cùng dimension, vô hiệu hóa eligibility theo fingerprint và enqueue lại. Khi đổi dimension, phải rebuild cấu trúc vector tương thích hoặc từ chối cấu hình với lỗi rõ ràng; không ghi vector khác dimension vào bảng cũ. Trong cả hai trường hợp, FTS vẫn là đường dự phòng.

Rollout host thật là bước riêng sau khi implementation được kiểm tra:

1. Chạy toàn bộ migration, backfill và crash/recovery trên database tạm hoặc bản sao bằng SQLite backup API.
2. Ghi snapshot an toàn và xác minh bản backup trước live migration.
3. Dừng các client cũ có thể đọc vector không qua freshness gate, rồi cập nhật các reader/writer/worker cùng đợt. Kiểm kê MCP, CLI, dashboard và scheduler; không giả định chỉ có một process.
4. Chạy migration, khởi động đúng một lịch worker đã chọn và theo dõi backfill.
5. Smoke MCP search và dashboard trên dữ liệu canary được chấp thuận; đối soát nguồn, metadata, vector và jobs.

Không rollback bằng cách chạy binary cũ lên database mới. Nếu cần dừng tính năng, giữ phiên bản code tương thích schema, tắt semantic search và tiếp tục FTS trong khi sửa lỗi. Phương án restore phải cân nhắc các memory ghi sau snapshot; không tự restore làm mất chúng.

## 9. Quan sát được

Status phải có số liệu theo từng loại entity và database/profile:

- `source_total`, `eligible`, `missing`, `stale`, `unverified`, `orphan`.
- Jobs pending, claimed, retry, blocked; tuổi job chờ lâu nhất và lỗi gần nhất.
- Model fingerprint active, lần reconcile gần nhất, chế độ hybrid/FTS-only/degraded.

Các nhóm coverage phải có định nghĩa rõ và không đếm trùng khi cộng tổng source. Job counts là chiều đo riêng, không cộng vào coverage. Không log nội dung memory, vector đầy đủ hoặc credential để chẩn đoán.

Memory store thành công có nghĩa nội dung đã commit. Nếu vector còn pending, phản hồi/status phải phân biệt rõ, không báo semantic index ready. Giữ tương thích envelope MCP hiện có nếu có thể; mọi field mới cần kiểm thử consumer.

## 10. Trình tự triển khai và quyền sở hữu

| Bước | Phạm vi | Bằng chứng hoàn thành |
| --- | --- | --- |
| 1 | Xác minh upstream, dependency, toàn bộ writer/reader/scheduler; tái hiện lỗi baseline | Ghi commit chính xác, repro isolated và inventory các entry point. |
| 2 | Chốt schema, eligibility predicate, trigger order, lease và generation contract | Test nhỏ chứng minh enqueue atomic và prefilter top-k trên dependency được hỗ trợ. |
| 3 | Migration, outbox, worker, vector publish và metadata | Test rollback, retry, concurrency và crash/restart đạt. |
| 4 | Tích hợp upsert, MCP, dashboard, reindex, scheduled invocation | Không còn đường đọc hoặc ghi vector bỏ qua hợp đồng mới. |
| 5 | Status, reconcile, backfill, tài liệu vận hành | Dry-run chỉ đọc; backfill resumable; số liệu phản ánh đúng trạng thái. |
| 6 | Independent review và kiểm thử tích hợp | Reviewer kiểm tra artifacts và cố tái hiện stale result, không chỉ đọc báo cáo implementer. |
| 7 | Đề xuất rollout cụ thể | Có backup, danh sách client phải restart, smoke và cách phục hồi; host mutation chỉ khi được giao. |

Giữ schema, triggers, worker và publish dưới một owner vì cùng một hợp đồng transaction. Nếu dùng agent song song, reviewer hoặc owner search có thể làm độc lập sau khi hợp đồng dữ liệu đã chốt. Tuân thủ AGENTS.md hiện hành về teamplay/agy khi thực sự triển khai; không tự dispatch từ tài liệu plan này.

## 11. Bộ kiểm thử bắt buộc

| Tình huống | Kết quả phải chứng minh |
| --- | --- |
| Thêm memory, model hoạt động | Nội dung và job commit; vector thành eligible sau worker run. |
| Embedding lỗi rồi phục hồi | Nội dung/FTS còn nguyên; job retry; tự thành eligible mà không reindex thủ công. |
| Sửa `v1` thành `v2` | Ngay sau commit, truy vấn snapshot mới không dùng vector `v1`; FTS trả nội dung `v2`. |
| Sửa tiếp thành `v3` khi worker tính `v2` | Kết quả `v2` bị loại; job `v3` còn và cuối cùng publish đúng. |
| Rollback ghi nội dung | Revision và enqueue rollback cùng nhau, không có job mồ côi. |
| Crash sau claim, trước publish | Lease hết hạn, worker khác xử lý được; không mất job. |
| Crash giữa delete/insert vector/metadata/ack | Transaction rollback hoặc commit đủ; không có trạng thái ready giả. |
| Hai worker, lease cũ hết hạn | Chỉ token hiện hành được publish; worker cũ không ack hoặc xóa job mới. |
| Xóa, đổi ID, xóa rồi tạo lại cùng ID | Không hồi sinh vector cũ, không mất job cho entity mới. |
| Restore trong lúc worker đang embed | Generation cũ bị từ chối kể cả khi số revision trùng; reconcile khôi phục tiến độ. |
| Consolidation, migration, SQL write hợp lệ | Trigger chung tạo job/invalidation đúng; không phụ thuộc chỉ vào hàm upsert. |
| Legacy vector chưa có metadata | Không eligible trước backfill; backfill chạy lại không tạo duplicate và chịu được update đồng thời. |
| Stale vectors gần query hơn vector hợp lệ | Top-k lấy đủ candidate hợp lệ nếu có; không bị thiếu do lọc sau KNN. |
| Source đổi giữa eligibility và hydrate | Mỗi response dùng snapshot nhất quán; không ghép vector cũ với nội dung mới. |
| Thiếu/tắt sqlite-vec hoặc model không tải được | FTS và lưu nội dung vẫn hoạt động; status phản ánh suy giảm; không busy-loop. |
| Đổi model hoặc dimension | Không trộn embedding space; cấu hình không hỗ trợ bị từ chối rõ ràng, không làm mất nội dung. |
| Embedding từng trả `None`, model phục hồi trong cùng process | Lỗi cache không ngăn retry thành công. |
| Turn queue rỗng hoặc đang debounce | Vector jobs vẫn được phục vụ trong lịch chạy; không bị nhánh exit sớm bỏ qua. |
| Custom DB và dashboard profile | Job, metadata, query và model config dùng đúng database; không rò giữa profile. |
| MCP và dashboard integration | Cùng loại stale candidate đều bị loại trên cả hai surface; source/graph semantics giữ nguyên. |

Parameterize các test cốt lõi cho facts, episodes và learnings. Dùng SQLite thật cùng extension thật cho transaction/KNN tests; chỉ mock embedding, đồng hồ hoặc failure boundary khi cần kiểm soát. Có ít nhất một canary với model thật và MCP stdio trên database disposable; thiếu model thì ghi rõ phần chưa được chứng minh.

Đo thời gian source write với embedder bị chặn có kiểm soát để chứng minh request không đợi embedding. Kiểm tra p95 search và thời gian drain trên dataset cố định so với baseline. Không tự gán con số cải thiện khi chưa đo.

## 12. Definition of done và bàn giao

Hoàn tất implementation khi:

- Source thay đổi là vector cũ mất eligibility; không có reader còn dùng đường không lọc freshness.
- Tạo/sửa/xóa memory, rollback, worker crash, lease hết hạn và restore đều giữ đúng các nguyên tắc ở mục 3.
- Sau khi model phục hồi và worker drain xong trên fixture ổn định, mọi source được yêu cầu index có vector đúng revision/model/generation; missing, stale và unverified bằng 0; không còn job bị bỏ sót.
- Tất cả đường ghi được kiểm kê có enqueue/invalidation; embedding nằm ngoài write lock.
- Migration, reindex và reconcile resumable; dry-run không ghi DB, không tự tải model và không thay runtime.
- Có evidence từ source rows, vector metadata, job receipts/trạng thái, query thực tế và test results. Không chỉ dùng lời báo thành công của worker.
- Báo cáo tách rõ code/test đã hoàn tất, live rollout đã thực hiện hay chưa, và những giới hạn chưa kiểm chứng.

Agent bàn giao commit hoặc diff, kết quả tests, repro trước/sau, số liệu coverage trên fixture, hướng rollout và recovery. Không sửa database thật hoặc MCP config để làm đẹp báo cáo nghiệm thu.

Prompt ngắn để Kai giao agent sau khi quyết định triển khai:

> Triển khai theo VECTOR_INDEX_RELIABILITY_PLAN.md trong agy-memory-engine. Kiểm tra baseline đã merge trước khi sửa. Giữ nội dung memory, FTS, taxonomy và graph semantics. Ưu tiên atomic outbox, loại vector stale trước top-k, publish có revision/generation/lease fencing, và retry tự phục hồi. Làm trên branch/worktree phù hợp, kiểm thử database disposable, lấy independent review và báo bằng chứng. Chưa migrate/reindex database thật, đổi MCP config hoặc cài scheduler trên máy nếu chưa được giao bước rollout.
