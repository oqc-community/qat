from qat.purr.integrations.features import OpenPulseFeatures


def test_openpulsefeatures():
    opf = OpenPulseFeatures()
    assert len(opf.ports) == 0
