import unittest

from chomikgrad import verify_greedy_candidates


class SpeculativeGenerationTests(unittest.TestCase):
    def test_all_matching_draft_tokens_are_accepted(self) -> None:
        result = verify_greedy_candidates([3, 4, 5], [3, 4, 5])

        self.assertEqual(result.emitted_tokens, (3, 4, 5))
        self.assertEqual(result.accepted_draft_tokens, 3)

    def test_target_token_replaces_first_mismatch(self) -> None:
        result = verify_greedy_candidates([3, 9, 5], [3, 4, 7])

        self.assertEqual(result.emitted_tokens, (3, 4))
        self.assertEqual(result.accepted_draft_tokens, 1)

    def test_empty_or_unequal_blocks_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            verify_greedy_candidates([], [])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            verify_greedy_candidates([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
