"""Energy comparisons must weight power by time and reject partial measurements."""
import importlib.util
from pathlib import Path
import unittest

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('power_publication',REPO/'scripts/publish-exl3-power-comparison.py')
publication=importlib.util.module_from_spec(spec);spec.loader.exec_module(publication)


class PowerEnergyAccounting(unittest.TestCase):
    def cases(self):
        return [dict(card_energy_j=100.,batch_wall_s=1.,completion_tokens=2048 if i<54 else 1024) for i in range(70)]

    def test_time_weighted_power_and_full_output_energy(self):
        cases=self.cases();cases[-1].update(card_energy_j=800.,batch_wall_s=4.)
        result=publication.energy(cases)
        self.assertEqual(result['measured_wave_wall_s'],73.)
        self.assertEqual(result['card_energy_j'],7700.)
        self.assertAlmostEqual(result['mean_card_power_w'],7700/73)
        self.assertAlmostEqual(result['card_energy_wh'],7700/3600)
        self.assertAlmostEqual(result['card_j_per_output_token_including_prefill'],7700/(124*1024))
        self.assertAlmostEqual(result['output_tokens_per_s_including_prefill'],(124*1024)/73)
        self.assertAlmostEqual(result['output_tokens_per_wh_including_prefill'],(124*1024)*3600/7700)
        self.assertAlmostEqual(result['output_tokens_per_s_per_measured_watt_including_prefill'],
            result['output_tokens_per_s_including_prefill']/result['mean_card_power_w'])
        self.assertAlmostEqual(result['output_tokens_per_s_per_measured_watt_including_prefill'],
            1/result['card_j_per_output_token_including_prefill'])

    def test_incomplete_or_invalid_measurements_rejected(self):
        for invalid in ('partial','negative','nonfinite','missing_output'):
            with self.subTest(invalid=invalid):
                cases=self.cases()
                if invalid=='partial':cases.pop()
                elif invalid=='negative':cases[0]['card_energy_j']=-1.
                elif invalid=='nonfinite':cases[0]['card_energy_j']=float('nan')
                else:cases[0]['completion_tokens']-=1
                with self.assertRaises(AssertionError):publication.energy(cases)
