from absl.testing import absltest

from openwsn.hdlc import HDLC_FLAG, Hdlc


class HdlcTestCase(absltest.TestCase):

    def test_loopback(self):
        expected_data = b"\x65\x41\x45\x05"
        frame = Hdlc.hdlcify(expected_data)
        actual_data = Hdlc.dehdlcify(frame)
        self.assertEqual(expected_data, actual_data)

    def test_set_dag_root(self):
        frame = (
            b"\x7e\x52\x59\xbb\xbb\x00\x00\x00\x00\x00\x00\x01\xde\xad\xbe\xef"
            b"\xca\xfe\xde\xad\xbe\xef\xca\xfe\xde\xad\xbe\xef\xa7\xd9\x7e")
        data = Hdlc.dehdlcify(frame)

    def test_incorrect_first_byte(self):
        frame = b"\x65\x41\x45\x05" + HDLC_FLAG
        with self.assertRaises(ValueError):
            Hdlc.dehdlcify(frame)

    def test_incorrect_last_byte(self):
        frame = HDLC_FLAG + b"\x65\x41\x45\x05"
        with self.assertRaises(ValueError):
            Hdlc.dehdlcify(frame)

    def test_too_short_frame(self):
        frame = HDLC_FLAG + HDLC_FLAG
        with self.assertRaises(ValueError):
            Hdlc.dehdlcify(frame)

    def test_incorrect_crc(self):
        frame = HDLC_FLAG + b"\x65\x41\x45\x05" + HDLC_FLAG
        with self.assertRaises(ValueError):
            Hdlc.dehdlcify(frame)


if __name__ == "__main__":
    absltest.main()
