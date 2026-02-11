# Q&A Filter Report

**Generated**: 2026-02-11T00:30:39.602703Z

## Summary

| Metric | Count |
|--------|-------|
| Files Processed | 1579 |
| Total Q&A Pairs | 16,278 |
| Passed | 14,621 (89.8%) |
| Rejected | 1,657 (10.2%) |

## Rejection Reasons

| Reason | Count | % of Rejected |
|--------|-------|---------------|
| answer_length | 1275 | 76.9% |
| self_referential | 201 | 12.1% |
| question_diversity | 91 | 5.5% |
| question_type | 83 | 5.0% |
| generic_answer | 7 | 0.4% |

## Warnings

- -025: All Q&A pairs filtered out
- -m: All Q&A pairs filtered out
- 00-00-intro-04_maintena: Only 1 question type(s): safety
- 00-03_torque-s: Only 1 question type(s): factual
- 00-04_torque-s: Only 1 question type(s): factual
- 00-06_torque-s: Only 1 question type(s): factual
- 00-09_torque-s: Only 1 question type(s): factual
- 00-14_maintena: Only 2 Q&A pairs remaining
- 00-14_maintena: Only 1 question type(s): factual
- 00-torque-specs-004: Only 1 question type(s): factual
- 00-torque-specs-005: Only 1 question type(s): factual
- 00-torque-specs-006: Only 1 question type(s): factual
- 00-torque-specs-007: Only 1 question type(s): factual
- 00-torque-specs-008: Only 1 question type(s): factual
- 00-torque-specs-013: Only 1 Q&A pairs remaining
- 00-torque-specs-013: Only 1 question type(s): factual
- 00-torque-specs-015: Only 1 Q&A pairs remaining
- 00-torque-specs-015: Only 1 question type(s): factual
- 00-torque-specs-016: Only 1 Q&A pairs remaining
- 00-torque-specs-016: Only 1 question type(s): procedural

## Sample Rejections

### self_referential (5 samples)

1. Q: "What component is being removed according to the t..." A: "Roll pins are being removed, specifically the 2nd ..."
2. Q: "What type of bearing is being installed in this im..." A: "A new genuine bearing is being installed...."
3. Q: "What type of bearing construction is shown in this..." A: "This appears to be a ball bearing with multiple st..."

### answer_length (5 samples)

1. Q: "What is the bearing part number shown in the circu..." A: "1 09 0070..."
2. Q: "What country is marked on the bearing component?..." A: "GERMANY..."
3. Q: "What is the complete bearing identification number..." A: "1 09 0070..."

### question_diversity (5 samples)

1. Q: "What should be used between the lifting arm and ve..." A: "Apply rubber block of lifting arm on the rear perp..."
2. Q: "What components are shown in the rear towing eyes ..." A: "The diagram shows the rear towing eyes location on..."
3. Q: "What is the order number for the Service Indicator..." A: "The adapter order number is 62 1 140...."

### generic_answer (5 samples)

1. Q: "What information is typically found in a tightenin..." A: "A tightening torques section typically contains sp..."
2. Q: "Do hex screws or Torx bolts generally require high..." A: "Hex screws generally require higher torque values ..."
3. Q: "What torque specification applies to Bosch and Mag..." A: "The specification shows M8 screw type but the torq..."

### question_type (5 samples)

1. Q: "How does the torque angle differ between M10 and M..." A: "M10 main bearing screws require 70° torque angle, ..."
2. Q: "Which engine types use the S38/S14/M40/M42/M43/M44..." A: "These engine types use M10 screws with no specific..."
3. Q: "What is the difference in torque angle between eng..." A: "M10 main bearing screws require a 70° torque angle..."
