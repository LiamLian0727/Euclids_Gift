"""Regression tests for answer extraction and grading on Euclid30K."""

import unittest

from grader import grade_answer


class GraderTest(unittest.TestCase):
    def test_equivalence(self):
        cases = [
            (r"\boxed{\frac{1}{2}}", "0.5"),
            (r"\boxed{\sqrt{8}}", r"$2\sqrt{2}$"),
            ("0.333333", r"\frac{1}{3}"),
            (r"50\%", "0.5"),
            ("4, 3", "4; 3"),
            (r"\boxed{6} \text{ and } \boxed{8}", "6; 8"),
            ("3,000", "3000"),
            (r"30^{\circ}", "30°"),
            ("**Final answer:** 4", "4"),
            ("**Final answer**: 4", "4"),
            ("Final answer:\n4", "4"),
            ("The answer is B.", "B"),
            ("The answer is $B$.", "B"),
            ("最终答案是 4", "4"),
            (r"\boxed{\text{B}}", "b"),
        ]
        for response, reference in cases:
            with self.subTest(response=response, reference=reference):
                self.assertTrue(grade_answer(response, reference).correct)

    def test_wrong_or_ambiguous_answers(self):
        cases = [
            ("2.01", "2"),
            ("1.00000001", "1.0"),
            ("0", "1e-10"),
            ("3; 4", "4; 3"),
            (r"\boxed{8} and \boxed{6}", "6; 8"),
            (r"An intermediate value is \boxed{6}. Thus \boxed{8}.", "6; 8"),
            ("(4,3)", "(3,4)"),
            ("4,4", "4"),
            ("4 or 5", "4"),
            ("9=4", "4"),
            ("4x=4", "4"),
            (r"4\mathrm{cm}^2", "4m^2"),
            (r"72\mathrm{in}", "72cm"),
            (r"<think>\boxed{4}</think>\boxed{5}", "4"),
            ("We get \\boxed{4}.\nFinal answer: 5", "4"),
            (r"\boxed{AB}", "BA"),
            ("", "1"),
        ]
        for response, reference in cases:
            with self.subTest(response=response, reference=reference):
                self.assertFalse(grade_answer(response, reference).correct)

    def test_public_reference_notation(self):
        cases = [
            ("34.6", r"$$\number{34.6}$$"),
            ("5", r"$5{}^\circ $"),
            (r"36\pi", r"$36\\pi\\ \mathrm{m}^2$"),
            (r"\frac{10\sqrt{3}}{3}", r"$\frac{10\sqrt{3}}{3}\cdot $"),
            ("(-3,4)", r"$$\left (-3,4\right )$$"),
            ("609.47", r"Volume $=609.47 \mathrm{~cm}^{3}$"),
            ("1209.84", "S.A. $=1209.84$"),
            ("60", "60 º"),
            ("y=|x|", r"$$y=\left \lvert x\right \rvert $$"),
        ]
        for response, reference in cases:
            with self.subTest(response=response, reference=reference):
                self.assertTrue(grade_answer(response, reference).correct)


if __name__ == "__main__":
    unittest.main()
