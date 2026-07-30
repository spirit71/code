#!/usr/bin/env python3
from __future__ import annotations

"""把 evaluation_summary.json 导出成一个简单的 Excel 工作簿。"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED


def col_letter(idx: int) -> str:
    result = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        result = chr(65 + rem) + result
    return result


def excel_text_cell(value: str, style_id: int = 0) -> str:
    return f'<c s="{style_id}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def excel_number_cell(value: float, style_id: int = 0) -> str:
    return f'<c s="{style_id}"><v>{value}</v></c>'


def make_sheet_xml(rows: list[list[tuple[str, object]]]) -> str:
    # 这里不依赖 pandas/openpyxl，直接手写 xlsx 所需的 worksheet xml。
    # 这样即使环境比较干净，也能稳定导出 Excel。
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, (cell_type, value) in enumerate(row, start=1):
            ref = f"{col_letter(col_idx)}{row_idx}"
            if cell_type == "text":
                cell_xml = excel_text_cell(str(value), style_id=1 if row_idx == 1 else 0)
            elif cell_type == "percent":
                cell_xml = excel_number_cell(float(value), style_id=2)
            elif cell_type == "number":
                cell_xml = excel_number_cell(float(value), style_id=0)
            else:
                raise ValueError(f"Unsupported cell type: {cell_type}")
            cells.append(cell_xml.replace("<c ", f'<c r="{ref}" ', 1))
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    last_col = col_letter(max(len(row) for row in rows)) if rows else "A"
    last_row = len(rows) if rows else 1
    dimension = f"A1:{last_col}{last_row}"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="{dimension}"/>
  <sheetViews>
    <sheetView workbookViewId="0"/>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <sheetData>
    {''.join(sheet_rows)}
  </sheetData>
</worksheet>
'''


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(
            f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    {''.join(sheets)}
  </sheets>
</workbook>
'''


def workbook_rels_xml(sheet_names: list[str]) -> str:
    rels = []
    for idx, _ in enumerate(sheet_names, start=1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{len(sheet_names) + 1}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        f'Target="styles.xml"/>'
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>
'''


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {''.join(overrides)}
</Types>
'''


def root_rels_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
'''


def styles_xml() -> str:
    # 三种样式就够当前需求: 普通文本、表头粗体、数值保留两位小数。
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1">
    <numFmt numFmtId="164" formatCode="0.00"/>
  </numFmts>
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>
'''


def core_xml() -> str:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Evaluation Summary Export</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>
'''


def app_xml(sheet_names: list[str]) -> str:
    titles = "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name in sheet_names)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>{len(sheet_names)}</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="{len(sheet_names)}" baseType="lpstr">
      {titles}
    </vt:vector>
  </TitlesOfParts>
</Properties>
'''


def build_rows(stage_name: str, dataset_recalls: dict[str, dict[str, float]]) -> list[list[tuple[str, object]]]:
    # 把某个 stage 的多数据集结果整理成二维表，后面直接写进 Excel worksheet。
    rows = [[("text", "Dataset"), ("text", "Stage"), ("text", "R@1 (%)"), ("text", "R@5 (%)"), ("text", "R@10 (%)"), ("text", "R@20 (%)")]]
    for dataset, recalls in dataset_recalls.items():
        rows.append([
            ("text", dataset),
            ("text", stage_name),
            ("percent", recalls.get("R@1", 0.0) * 100.0),
            ("percent", recalls.get("R@5", 0.0) * 100.0),
            ("percent", recalls.get("R@10", 0.0) * 100.0),
            ("percent", recalls.get("R@20", 0.0) * 100.0),
        ])
    return rows


def export_summary(json_path: Path, output_path: Path) -> None:
    # 导出 3 个 sheet:
    # 1. Summary: val + test 汇总
    # 2. Validation: 只看验证集
    # 3. Test: 只看测试集
    data = json.loads(json_path.read_text(encoding="utf-8"))

    summary_rows = [[("text", "Dataset"), ("text", "Stage"), ("text", "R@1 (%)"), ("text", "R@5 (%)"), ("text", "R@10 (%)"), ("text", "R@20 (%)")]]
    for stage_name in ["val", "test"]:
        for dataset, recalls in data.get(stage_name, {}).items():
            summary_rows.append([
                ("text", dataset),
                ("text", stage_name),
                ("percent", recalls.get("R@1", 0.0) * 100.0),
                ("percent", recalls.get("R@5", 0.0) * 100.0),
                ("percent", recalls.get("R@10", 0.0) * 100.0),
                ("percent", recalls.get("R@20", 0.0) * 100.0),
            ])

    sheets = [
        ("Summary", summary_rows),
        ("Validation", build_rows("val", data.get("val", {}))),
        ("Test", build_rows("test", data.get("test", {}))),
    ]
    sheet_names = [name for name, _ in sheets]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml(sheet_names))
        zf.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(sheet_names))
        zf.writestr("xl/styles.xml", styles_xml())
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", make_sheet_xml(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export evaluation summary JSON to Excel (.xlsx).")
    parser.add_argument("input_json", type=Path, help="Path to evaluation_summary_epoch_XX.json")
    parser.add_argument("-o", "--output", type=Path, help="输出 xlsx 路径；默认与 json 同名。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_json = args.input_json.resolve()
    if not input_json.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    output_path = args.output.resolve() if args.output else input_json.with_suffix(".xlsx")
    export_summary(input_json, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
