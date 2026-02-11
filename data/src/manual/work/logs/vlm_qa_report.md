# VLM Dataset Validation Report

**Generated**: 2026-02-10T19:31:13.832845
**Status**: ✅ PASSED

## Summary

| Metric | Train | Val | Total |
|--------|-------|-----|-------|
| Q&A Pairs | 11,004 | 1,404 | 12,408 |
| Unique Images | 10,997 | 1,379 | 12,376 |
| Text-only Q&A | 7 | 25 | 32 |

## Distribution by Section

| Section | Train | Val |
|---------|-------|-----|
|  | 826 | 86 |
| 00 | 1,242 | 143 |
| 11 | 443 | 61 |
| 12 | 715 | 80 |
| 13 | 547 | 67 |
| 16 | 99 | 23 |
| 17 | 86 | 8 |
| 18 | 72 | 12 |
| 1990 | 936 | 108 |
| 21 | 73 | 10 |
| 23 | 817 | 98 |
| 25 | 183 | 31 |
| 26 | 151 | 20 |
| 31 | 517 | 71 |
| 33 | 186 | 28 |
| 34 | 482 | 54 |
| 35 | 91 | 17 |
| 36 | 105 | 22 |
| 41 | 1,131 | 122 |
| 51 | 1,059 | 131 |
| 52 | 39 | 11 |
| 54 | 158 | 14 |
| 62 | 190 | 32 |
| 63 | 49 | 9 |
| 64 | 295 | 39 |
| 65 | 325 | 38 |
| 72 | 89 | 23 |
| 97 | 18 | 7 |
| bosch | 73 | 6 |
| reference | 0 | 8 |
| techspec | 7 | 25 |

## Distribution by Source Type

| Source Type | Train | Val |
|-------------|-------|-----|
| ecu_technical | 73 | 6 |
| electrical_manual | 936 | 108 |
| html_specs | 7 | 25 |
| service_manual | 9,378 | 1,180 |
| unknown | 610 | 85 |

## Critical Errors

None

## Warnings

None

## Sample Q&A Pairs

### Training Set

**1.** Q: What does callout number 2 indicate in the glove box diagram?...
   A: The trim panel that needs to be taken off after unscrewing the screw....

**2.** Q: What is the difference in torque values between the fuel injector to intake manifold and the couplin...
   A: The coupling nut requires 25 Nm while the fuel injector to intake manifold requires 10 Nm, a differe...

**3.** Q: What country is marked on the bearing?...
   A: The bearing is marked with GERMANY....

**4.** Q: What is the gas filler lock motor's location in the vehicle?...
   A: The gas filler lock motor is located on the right side of the trunk, as indicated in Figure 3....

**5.** Q: What type of nuts should be replaced when working on the exhaust pipes?...
   A: Replace self-locking nuts....

### Validation Set

**1.** Q: What does callout number 3 identify in this sound system diagram?...
   A: Speakers - rear...

**2.** Q: What manual section covers manual transmission specifications?...
   A: Section 23-1 Manual Transmission, subsection 23 00 Transmission in general...

**3.** Q: Which engine types apply to the main bearing cap inclined bolts specifications?...
   A: M60/1/M60/2/M62/M70/S70/M73...

**4.** Q: What wire color designation is shown for the connection to the START circuit?...
   A: The START circuit shows a wire color designation of 5 BK/GN....

**5.** Q: What is the page number and section title for this component locations reference?...
   A: This is page 7000-7 from the Component Locations Views section....
