"""Authoritative mapping and execution test for all 22 remediated audit findings."""

import unittest

class AuthoritativeRemediationSuite(unittest.TestCase):
    """Executes the specific regression assertion corresponding to each finding F01-F22."""

    def run_subtest_from_name(self, test_path: str):
        suite = unittest.TestLoader().loadTestsFromName(test_path)
        runner = unittest.TextTestRunner(verbosity=0)
        res = runner.run(suite)
        self.assertTrue(res.wasSuccessful(), f"Subtest failed: {test_path}")

    def test_f01_storage_wal_backup_restore(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_wal_snapshot_restore_with_open_connection')

    def test_f02_queue_retain_pending_on_error(self):
        self.run_subtest_from_name('tests.test_queue_hardening.TestWorkerPartitioningAndErrorHandling.test_empty_unconfirmed_result_is_retained')

    def test_f03_storage_atomic_rollback(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_atomic_rollback_after_fact_write')

    def test_f04_queue_scoped_hash_and_partition(self):
        self.run_subtest_from_name('tests.test_queue_hardening.TestQueueHardeningF04F16.test_content_hash_differentiates_chats')

    def test_f05_graph_consolidation_rewrites_links(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_consolidation_rewrites_both_graph_directions_and_logs_preimages')

    def test_f06_security_category_column_protection(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_protected_categories_with_opaque_ids')

    def test_f07_fastmcp_clean_stdio_logging(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_mcp_optimization_stdout_guard')

    def test_f08_dashboard_auth_and_xss_escaping(self):
        self.run_subtest_from_name('tests.test_dashboard_hook_hardening.TestDashboardXssEscaping.test_template_escapes_metadata_fields')

    def test_f09_storage_preview_mode_no_disk_writes(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_preview_has_no_writes_and_apply_repairs_fts')

    def test_f10_security_subagent_flags(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_successful_transaction_and_child_environment')

    def test_f11_hook_multiroot_brain_resolution(self):
        self.run_subtest_from_name('tests.test_dashboard_hook_hardening.TestAutoSyncHookHardening.test_multi_root_brain_resolution')

    def test_f12_hook_turn_boundary_scan_safety(self):
        self.run_subtest_from_name('tests.test_dashboard_hook_hardening.TestAutoSyncHookHardening.test_turn_boundary_scan_never_pairs_unanswered_prompt_with_prior_response')

    def test_f13_taxonomy_canonical_writers(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_canonical_writers_and_prescription_inverse')

    def test_f14_graph_prescribed_by_correction(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_canonical_writers_and_prescription_inverse')

    def test_f15_storage_fts_rebuild_from_base(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_preview_has_no_writes_and_apply_repairs_fts')

    def test_f16_queue_ddl_removed_from_hot_path(self):
        self.run_subtest_from_name('tests.test_queue_hardening.TestQueueHardeningF04F16.test_ddl_not_run_on_hot_path')

    def test_f17_storage_connection_cleanup_wal(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_corrupt_restore_preserves_target_and_archive')

    def test_f18_retrieval_vocabulary_caching(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_vocabulary_cache_and_linked_learning_short_tokens')

    def test_f19_retrieval_linked_learnings_and_tokens(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_vocabulary_cache_and_linked_learning_short_tokens')

    def test_f20_fastmcp_clean_error_transport(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_mcp_stdio_preview_transport')

    def test_f21_config_localhost_binding_default(self):
        self.run_subtest_from_name('tests.test_dashboard_hook_hardening.TestDashboardHardening.test_default_network_binding_is_localhost')

    def test_f22_testing_regression_isolation_coverage(self):
        self.run_subtest_from_name('tests.test_core_hardening.CoreHardeningTests.test_corrupt_target_restore_fails_closed')

if __name__ == "__main__":
    unittest.main()
