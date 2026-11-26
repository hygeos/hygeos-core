import pytest

from hygeos_core.geo.product_name import get_pattern, retrieve_product, get_level

@pytest.mark.parametrize(['sensor', 'example_prod'], [
    ('LANDSAT-8-OLI'     ,'LC08_L1GT_029030_20151209_20160131_01_RT'),
    ('LANDSAT-9-OLI'     ,'LC09_L1TP_014034_20220618_20230411_02_T1'),
    ('SENTINEL-3-OLCI-FR','S3A_OL_1_EFR____20150101T102500_20150101T110900_20150101T113000_2640_030_215_4520_MAR_O_NR_001.SEN3'),
    ('SENTINEL-3-OLCI-RR','S3A_OL_1_ERR____20150101T102500_20150101T110900_20150101T113000_2640_030_215______LN1_O_NT_001.SEN3'),
    ('SENTINEL-3-SLSTR'  ,'S3A_SL_2_WST____20150101T102500_20150101T114000_20150101T124000_4500_030_215______MAR_O_NR_001.SEN3'),
    ('SENTINEL-3-SRAL'   ,'S3A_SR_0_SRA____20150101T102500_20150101T114000_20150101T115000_4500_030_215______SVL_O_NR_TST.SEN3'),
    ('ENVISAT-MERIS'     ,'MER_RR__1PRACR20080701_014028_000026402070_00003_33123_0000.N1'),
    ('ENVISAT-MERIS'     ,'MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1'),
    ('ENVISAT-MERIS-SAFE','ENV_ME_1_RRG____20071102T084033_20071102T092429_________________2636_063_050______DSI_R_NT____.SEN3'),
])
def test_identify_product(sensor, example_prod):
    assert sensor == get_pattern(example_prod)['Name']

@pytest.mark.parametrize('example_prod', ['LC08_L1GT_029030_20151209_20160131_01_RT'])
def test_retrieve_product(example_prod):
    pattern = get_pattern(example_prod)
    print(retrieve_product(example_prod, {'level': 'L2GS'}, pattern))

@pytest.mark.parametrize('example_prod', ['LC08_L1GT_029030_20151209_20160131_01_RT'])
def test_get_level(example_prod):
    pattern = get_pattern(example_prod)
    print(get_level(example_prod, pattern))