#!/usr/bin/env python3
# Safely add frozen five-trial phase support to carla/collect_data.py.

from pathlib import Path
import re
import shutil
import py_compile

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "carla" / "collect_data.py"
BACKUP = ROOT / "carla" / "collect_data.py.pre_trial_phase_backup"

TRIAL_BLOCK = """
# ============================================================
# FROZEN FINAL-TRIAL LATERAL PHASES
# ============================================================

TRIAL_PHASE_DEG = {
    1: -8.0,
    2: -4.0,
    3: 0.0,
    4: 4.0,
    5: 8.0,
}

"""

ARG_BLOCK = """
    parser.add_argument(
        "--trial",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help=(
            "Frozen controlled trajectory configuration. "
            "1=-8deg, 2=-4deg, 3=0deg, 4=+4deg, 5=+8deg. "
            "For the final sweep always provide --trial explicitly."
        ),
    )

"""


def find_add_argument_block(text, option):
    idx = text.find(option)
    if idx < 0:
        raise RuntimeError(f"Could not find argument {option!r}")

    start = text.rfind("parser.add_argument(", 0, idx)
    if start < 0:
        raise RuntimeError(f"Could not find parser.add_argument for {option!r}")

    pos = start
    depth = 0
    in_str = None
    escaped = False

    while pos < len(text):
        ch = text[pos]
        if in_str is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = pos + 1
                    while end < len(text) and text[end] in " \t":
                        end += 1
                    if end < len(text) and text[end] == "\n":
                        end += 1
                    return start, end
        pos += 1

    raise RuntimeError(f"Unterminated parser.add_argument block for {option!r}")


def insert_call_argument(text, marker, argument_line):
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError(f"Missing call marker: {marker!r}")

    call_idx = text.find("smooth_lateral_pose(", idx)
    if call_idx < 0:
        raise RuntimeError(f"No smooth_lateral_pose call after {marker!r}")

    pos = call_idx + len("smooth_lateral_pose")
    depth = 0
    in_str = None
    escaped = False

    while pos < len(text):
        ch = text[pos]
        if in_str is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    line_start = text.rfind("\n", 0, pos) + 1
                    closing_indent = text[line_start:pos]
                    insert = f"{closing_indent}    {argument_line}\n"
                    return text[:line_start] + insert + text[line_start:]
        pos += 1

    raise RuntimeError("Could not find smooth_lateral_pose call closing parenthesis.")


def find_writer_block(text, start_at, must_contain):
    idx = text.find("writer.writerow([", start_at)
    while idx >= 0:
        end = text.find("])", idx)
        if end < 0:
            raise RuntimeError("Unterminated writer.writerow block.")
        block = text[idx:end + 2]
        if all(token in block for token in must_contain):
            return idx, end + 2, block
        idx = text.find("writer.writerow([", end + 2)
    raise RuntimeError(f"Could not find writer block containing {must_contain}")


def main():
    if not TARGET.exists():
        raise RuntimeError(f"Missing target: {TARGET}")

    text = TARGET.read_text()

    if "TRIAL_PHASE_DEG = {" in text or '"--trial"' in text:
        raise RuntimeError(
            "Trial-phase support appears to already exist. No changes made."
        )

    shutil.copy2(TARGET, BACKUP)
    print("Backup created:", BACKUP)

    # 1. Frozen mapping.
    anchor = "POSE_ANGLE_TOL_DEG"
    aidx = text.find(anchor)
    if aidx < 0:
        raise RuntimeError("Could not find POSE_ANGLE_TOL_DEG anchor.")
    line_end = text.find("\n", aidx)
    insert_at = line_end + 1
    text = text[:insert_at] + "\n" + TRIAL_BLOCK + text[insert_at:]

    # 2. --trial argument after --lateral-drift.
    _, end = find_add_argument_block(text, '"--lateral-drift"')
    text = text[:end] + "\n" + ARG_BLOCK + text[end:]

    # 3. smooth_lateral_pose signature and equation.
    sig_old = re.compile(
        r"(def\s+smooth_lateral_pose\s*\(\s*"
        r"sample_id\s*,\s*"
        r"total_samples\s*,\s*"
        r"amplitude_m\s*,)(\s*\):)",
        flags=re.S,
    )
    text, n = sig_old.subn(
        r"\1\n    phase_offset_deg=0.0,\2",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError(
            f"Expected to patch smooth_lateral_pose signature once, got {n}."
        )

    amp_line = "    amplitude_m = float(amplitude_m)\n"
    if amp_line not in text:
        raise RuntimeError("Could not find amplitude conversion line.")
    text = text.replace(
        amp_line,
        amp_line + "    phase_offset_deg = float(phase_offset_deg)\n",
        1,
    )

    phase_old = re.compile(
        r"    phase\s*=\s*\(\s*"
        r"2\.0\s*\*\s*np\.pi\s*\*\s*progress\s*"
        r"\)",
        flags=re.S,
    )
    phase_new = (
        "    phase = (\n"
        "        2.0 * np.pi * progress\n"
        "        + np.deg2rad(phase_offset_deg)\n"
        "    )"
    )
    text, n = phase_old.subn(phase_new, text, count=1)
    if n != 1:
        raise RuntimeError(f"Expected to patch phase equation once, got {n}.")

    # 4. Main: trial -> phase.
    drift_line = "    lateral_drift_m = float(args.lateral_drift)\n"
    if drift_line not in text:
        raise RuntimeError("Could not find lateral_drift_m assignment in main.")

    main_insert = (
        "    trial_id = args.trial\n\n"
        "    if trial_id is None:\n"
        "        # Legacy/pilot compatibility only.\n"
        "        phase_offset_deg = 0.0\n"
        "    else:\n"
        "        phase_offset_deg = float(\n"
        "            TRIAL_PHASE_DEG[trial_id]\n"
        "        )\n\n"
    )
    text = text.replace(
        drift_line,
        drift_line + "\n" + main_insert,
        1,
    )

    # 5. Trial suffix in output folder.
    out_anchor = "    output_root = os.path.join(\n"
    oidx = text.find(out_anchor, text.find("def main():"))
    if oidx < 0:
        raise RuntimeError("Could not find output_root assignment.")

    trial_name_block = (
        "    if trial_id is not None:\n"
        "        experiment_name = (\n"
        "            f\"{experiment_name}_trial_{trial_id}\"\n"
        "        )\n\n"
    )
    text = text[:oidx] + trial_name_block + text[oidx:]

    # 6. Pass phase to both pose calls.
    text = insert_call_argument(
        text,
        "first_lateral_offset_m",
        "phase_offset_deg,",
    )
    text = insert_call_argument(
        text,
        ") = smooth_lateral_pose(\n\n                    sample_id,",
        "phase_offset_deg,",
    )

    # 7. Log trial + phase in CSV header.
    hstart, hend, hblock = find_writer_block(
        text,
        text.find("# RECORD"),
        ['"sample_id"', '"severity"', '"lateral_offset_m"'],
    )
    header_target = '                "severity",\n'
    if header_target not in hblock:
        raise RuntimeError("Could not locate severity field in CSV header.")

    hblock_new = hblock.replace(
        header_target,
        header_target
        + '                "trial_id",\n'
        + '                "lateral_phase_deg",\n',
        1,
    )
    text = text[:hstart] + hblock_new + text[hend:]

    # 8. Log values in CSV data row.
    dstart, dend, dblock = find_writer_block(
        text,
        hstart + len(hblock_new),
        ["sample_id,", "target_frame,", "weather_name,", "severity,"],
    )
    data_target = "                    severity,\n"
    if data_target not in dblock:
        raise RuntimeError("Could not locate severity value in CSV data row.")

    dblock_new = dblock.replace(
        data_target,
        data_target
        + '                    ("" if trial_id is None else trial_id),\n'
        + '                    f"{phase_offset_deg:.3f}",\n',
        1,
    )
    text = text[:dstart] + dblock_new + text[dend:]

    # 9. Print resolved trial.
    weather_anchor = "    weather = make_weather(\n"
    widx = text.find(weather_anchor, text.find("def main():"))
    if widx < 0:
        raise RuntimeError("Could not find make_weather call.")

    print_block = (
        "    print(\n"
        "        \"Trajectory trial:\",\n"
        "        \"legacy\" if trial_id is None else trial_id,\n"
        "    )\n"
        "    print(\n"
        "        \"Lateral phase offset:\",\n"
        "        f\"{phase_offset_deg:+.1f} deg\",\n"
        "    )\n\n"
    )
    text = text[:widx] + print_block + text[widx:]

    TARGET.write_text(text)

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception:
        shutil.copy2(BACKUP, TARGET)
        print("Syntax check FAILED. Original restored.")
        raise

    print()
    print("PATCH COMPLETE")
    print("Syntax check: PASS")
    print("Updated:", TARGET)
    print()
    print("Frozen mapping:")
    print("  trial 1 -> -8 deg")
    print("  trial 2 -> -4 deg")
    print("  trial 3 ->  0 deg")
    print("  trial 4 -> +4 deg")
    print("  trial 5 -> +8 deg")
    print()
    print("Rain/Night exploratory configuration: trial 3")
    print()
    print("Verification:")
    print("  python carla/collect_data.py --help | grep -A4 trial")
    print("  grep -nE 'TRIAL_PHASE_DEG|phase_offset_deg|trial_id|--trial' carla/collect_data.py")


if __name__ == "__main__":
    main()
