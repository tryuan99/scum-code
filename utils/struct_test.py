from absl.testing import absltest

from utils.struct import Struct, StructFieldType


class TestStruct(Struct):
    """Test struct."""

    @property
    def fields(self) -> dict[str, (StructFieldType, int)]:
        """Returns a dictionary mapping each field name to its size in bytes
        and the array length.
        """
        return {
            "char": (StructFieldType.CHAR, 1),
            "uint8": (StructFieldType.UINT8, 1),
            "int32_array": (StructFieldType.INT32, 4),
        }


class StructTestCase(absltest.TestCase):

    def setUp(self):
        self.test_struct = TestStruct(
            bytearray(
                b"\x01\xAC\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B"
                b"\x0C\x0D\x0E\x0F"))

    def test_get_single_element(self):
        self.assertEqual(self.test_struct.get("char"), b"\x01")
        self.assertEqual(self.test_struct.get("uint8"), 0xAC)

    def test_set_single_element(self):
        self.test_struct.set("char", b"\xFF")
        self.test_struct.set("uint8", 200)
        self.assertEqual(self.test_struct.get("char"), b"\xFF")
        self.assertEqual(self.test_struct.get("uint8"), 200)

    def test_get_array(self):
        self.assertEqual(self.test_struct.get("int32_array", index=0),
                         0x03020100)
        self.assertEqual(self.test_struct.get("int32_array"),
                         [0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C])

    def test_set_array(self):
        self.test_struct.set("int32_array", 1048576, index=2)
        self.assertEqual(self.test_struct.get("int32_array", index=2), 1048576)
        self.test_struct.set("int32_array",
                             [314159265, 27182818, -1000000, 141421])
        self.assertEqual(self.test_struct.get("int32_array"),
                         [314159265, 27182818, -1000000, 141421])

    def test_get_buffer(self):
        self.assertEqual(
            self.test_struct.get_buffer(),
            bytearray(
                b"\x01\xAC\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B"
                b"\x0C\x0D\x0E\x0F"))
        self.assertEqual(self.test_struct.get_buffer(start=12),
                         bytearray(b"\x0A\x0B\x0C\x0D\x0E\x0F"))
        self.assertEqual(self.test_struct.get_buffer(end=2),
                         bytearray(b"\x01\xAC"))

    def test_set_buffer(self):
        self.test_struct.set_buffer(bytearray(5), start=6)
        self.assertEqual(
            self.test_struct.get_buffer(),
            bytearray(
                b"\x01\xAC\x00\x01\x02\x03\x00\x00\x00\x00\x00\x09\x0A\x0B"
                b"\x0C\x0D\x0E\x0F"))

    def test_buffer(self):
        struct = TestStruct()
        struct.set("char", b"\x01")
        struct.set("uint8", 0xAC)
        struct.set("int32_array",
                   [0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C])
        self.assertEqual(struct.buffer, self.test_struct.buffer)

    def test_invalid_field(self):
        with self.assertRaises(ValueError):
            self.test_struct.get("error")


if __name__ == "__main__":
    absltest.main()
