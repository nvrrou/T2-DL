import ast
import json
import tempfile
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "Tarea 2.ipynb"


class NotebookRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = [
            ast.parse("".join(cell["source"]))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]

    def scope_for(self, tree):
        nodes = [
            node for node in tree.body
            if (isinstance(node, ast.Import)
                and any(alias.name == "re" for alias in node.names))
            or (isinstance(node, ast.FunctionDef)
                and node.name == "extraer_numero")
        ]
        scope = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), scope)
        return scope

    def test_each_cell_sorts_images_independently(self):
        for index, tree in enumerate(self.cells):
            with self.subTest(cell=index):
                key = self.scope_for(tree)["extraer_numero"]
                paths = list(map(Path, ["foto10.jpg", "pez.jpg", "foto2.jpg"]))
                self.assertEqual(list(map(key, paths)), [10, 0, 2])
                self.assertEqual([p.name for p in sorted(paths, key=key)],
                                 ["pez.jpg", "foto2.jpg", "foto10.jpg"])

    def test_empty_csv_retains_columns_used_by_summary(self):
        import pandas as pd

        for index, tree in enumerate(self.cells):
            with self.subTest(cell=index):
                assignment = next(
                    node for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "df" for t in node.targets)
                )
                scope = {"pd": pd, "registros": []}
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
                df = scope["df"]
                self.assertEqual(df["archivo"].nunique(), 0)
                self.assertTrue(df["etiqueta"].value_counts().empty)
                with tempfile.TemporaryDirectory() as directory:
                    csv = Path(directory) / "detecciones.csv"
                    df.to_csv(csv, index=False)
                    self.assertEqual(list(pd.read_csv(csv).columns),
                                     ["archivo", "etiqueta", "confianza", "x1", "y1", "x2", "y2"])


if __name__ == "__main__":
    unittest.main()
