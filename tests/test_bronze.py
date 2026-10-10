import csv
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utils.bronze import TABLES, convert, sha256_file, verify


class BronzeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / TABLES["transactions"][0]
        self.row = ["2022/09/01 00:02", "000", "00A", "000", "00A", "001.00",
                    "USD", "1.000", "USD", "line,one\nline two", "0"]
        with self.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(TABLES["transactions"][1])
            writer.writerows([self.row, self.row, ["", " NA ", "NULL", "1", "B", "0",
                                                   "", "-1", "EUR", "", "1"]])
        self.sha = sha256_file(self.source)

    def build(self, name="output"):
        destination = self.root / name
        convert(self.source, destination, "transactions", self.sha, batch_rows=2)
        return destination

    def test_round_trip_edge_cases_and_stable_duplicate_ids(self):
        first, second = self.build(), self.build("repeat")
        report = verify(self.source, first, "transactions")
        self.assertEqual(report["rows"], 3)
        a = pq.read_table(first).to_pydict()
        b = pq.read_table(second).to_pydict()
        self.assertEqual(a, b)
        self.assertEqual(len(set(a["transaction_id"])), 3)
        self.assertEqual(a["from_bank"], ["000", "000", " NA "])
        self.assertEqual(a["timestamp"][-1], "")
        self.assertEqual(a["payment_format"][0], self.row[9])
        self.assertEqual(a["amount_received"][0], "001.00")

    def test_existing_output_is_not_overwritten(self):
        output = self.build()
        with self.assertRaises(FileExistsError):
            convert(self.source, output, "transactions", self.sha)

    def test_field_corruption_is_detected(self):
        output = self.build()
        part = output / "part-00000.parquet"
        table = pq.read_table(part)
        table = table.set_column(1, table.schema.field(1), pa.array(["WRONG", "000"]))
        pq.write_table(table, part)
        with self.assertRaisesRegex(ValueError, "Original field mismatch"):
            verify(self.source, output, "transactions")

    def test_bad_row_is_rejected_instead_of_skipped(self):
        with self.source.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(["only one field"])
        with self.assertRaisesRegex(ValueError, "no row skipped"):
            convert(self.source, self.root / "bad", "transactions", sha256_file(self.source))


if __name__ == "__main__":
    unittest.main()
