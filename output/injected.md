# EARS Rule Injection Results

## Summary

Total matches found: 5
Rules injected: 5
Rules already exist: 0
Rules skipped due to limit: 0

## Air-Conditioning Gateway Function_Ver_1_6.txt

### Rule 9: 2-1. input list

**Status:** inject

**Match Score:** 0.840

#### EARS Rule Details (Ground Truth)
- **Target Object (O):** System
- **Defect Condition (C):** ECU A receives a request from the Server that requires input from ECU B and does not wait for the specified waiting time,
- **Expected Erroneous Response (R_esp):** ECU A shall wait for the specified waiting time before responding to the Server.
- **Applied Mutation:** llm_rewrite
- **Mutation Details:** LLM autonomous rewrite

**Match Type:** ecu_based

**Location:** Lines 80-87

**Injected Content:**

For details of CAN data and LIN data, refer to "SUZUKI LIN SPECIFICATION" and "SUZUKI CAN
SPECIFICATION."
ECU A receives a request from the Server that requires input from ECU B and does not wait for the specified waiting time,

**Original Context:**
For details of CAN data and LIN data, refer to "SUZUKI LIN SPECIFICATION" and "SUZUKI CAN
SPECIFICATION."

---

### Rule 18: 3-1. gateway function control

**Status:** inject

**Match Score:** 0.720

#### EARS Rule Details (Ground Truth)
- **Target Object (O):** System
- **Defect Condition (C):** ECU A does not strictly follow the sequence "1) ECU A send request 1 to ECU B; 2) ECU A wait for 500ms before send next request to ECU B"
- **Expected Erroneous Response (R_esp):** the system shall treat the request sequence as invalid and notify a timing issue.
- **Applied Mutation:** llm_rewrite
- **Mutation Details:** LLM autonomous rewrite

**Match Type:** ecu_based

**Location:** Lines 2-7

**Injected Content:**

In this chapter, the air-conditioning gateway control is explained in each section.
(1)control objective
  ・Based on the operation information in IDS and remote function, the drive of ventilated seat ECU and steering
heater ECU is controlled.
  ・The actual operation state and failure state are received from the ventilated seat ECU and the steering heater
ECU, and the display is instructed to the IDS (the customer is notified).
the drive of ventilated seat ECU does not strictly follow the sequence "1) the drive of ventilated seat ECU send request 1 to heater ECU; 2) the drive of ventilated seat ECU wait for 500ms before send next request to heater ECU"

**Original Context:**
In this chapter, the air-conditioning gateway control is explained in each section.
(1)control objective
  ・Based on the operation information in IDS and remote function, the drive of ventilated seat ECU and steering
heater ECU is controlled.
  ・The actual operation state and failure state are received from the ventilated seat ECU and the steering heater
ECU, and the display is instructed to the IDS (the customer is notified).

---

### Rule 11: 3-1-2. Gateway function (conversion from LIN communication to CAN communication)

**Status:** inject

**Match Score:** 0.720

#### EARS Rule Details (Ground Truth)
- **Target Object (O):** System
- **Defect Condition (C):** the specification omits the explicit step "ECU A send request to stop other communicate to ECU B" before process start
- **Expected Erroneous Response (R_esp):** the ECU A shall treat the process sequence as invalid and not proceed.
- **Applied Mutation:** llm_rewrite
- **Mutation Details:** LLM autonomous rewrite

**Match Type:** ecu_based

**Location:** Lines 1-5

**Injected Content:**

(1)control objective
  The operation state and failure state received from the ventilated seat ECU and the steering heater ECU are
displayed on the IDS.
(2)control overview
  ・LIN signals received from ventilated seat ECU and steering heater ECU are converted into CAN signals and
transmitted to IDS.
  ・The failure information received from the ventilated seat ECU and the steering heater ECU is displayed and
instructed to the IDS.
the specification omits the explicit step "The operation state and failure state received from the ventilated seat ECU and the steering heater ECU send request to stop other communicate to LIN signals received from ventilated seat ECU and steering heater ECU" before process start

**Original Context:**
(1)control objective
  The operation state and failure state received from the ventilated seat ECU and the steering heater ECU are
displayed on the IDS.
(2)control overview
  ・LIN signals received from ventilated seat ECU and steering heater ECU are converted into CAN signals and
transmitted to IDS.
  ・The failure information received from the ventilated seat ECU and the steering heater ECU is displayed and
instructed to the IDS.

---

## Demo_doc_01.txt

### Rule 26: 3.1. Remote control communication sequence

**Status:** inject

**Match Score:** 0.600

#### EARS Rule Details (Ground Truth)
- **Target Object (O):** System
- **Defect Condition (C):** the ECU A executes the step "ECU A send request to ECU B when ECU B is ready" without first performing the step "if ECU B is not ready, wait for n second"
- **Expected Erroneous Response (R_esp):** the ECU A shall reject the request sequence and log a timeout handling error.
- **Applied Mutation:** llm_rewrite
- **Mutation Details:** LLM autonomous rewrite

**Match Type:** ecu_based

**Location:** Lines 133-140

**Injected Content:**

Content
Ignition
logical conjunction
logical sum
Local Interconnect Network (in-vehicle LAN standard)
Controller Area Network (in-vehicle LAN standard)
Unified Diagnostic Services
electronic control unit
Information Display System
Heating,Ventilation,and Air Conditioning
Defroster(Defrost, Defroster outlet)
A mechanism consisting of only switches.
Data Communication Module
Enhanced I/O ECU
Battery Management Unit/Battery Management System
Device for monitoring and controlling (protecting) lithium ion battery
the Enhanced I/O ECU executes the step "Enhanced I/O ECU send request to ECU when ECU is ready" without first performing the step "if ECU is not ready, wait for n second"

**Original Context:**
Content
Ignition
logical conjunction
logical sum
Local Interconnect Network (in-vehicle LAN standard)
Controller Area Network (in-vehicle LAN standard)
Unified Diagnostic Services
electronic control unit
Information Display System
Heating,Ventilation,and Air Conditioning
Defroster(Defrost, Defroster outlet)
A mechanism consisting of only switches.
Data Communication Module
Enhanced I/O ECU
Battery Management Unit/Battery Management System
Device for monitoring and controlling (protecting) lithium ion battery

---

### Rule 22: 3-2. Judgment of remote air conditioning implementation

**Status:** inject

**Match Score:** 0.318

#### EARS Rule Details (Ground Truth)
- **Target Object (O):** System
- **Defect Condition (C):** the specification omits the dependency between the steps "ECU C accept the request with positive response to ECU B" and "ECU B response with positive response to ECU A"
- **Expected Erroneous Response (R_esp):** the ECU B shall treat the sequence as invalid and not proceed.
- **Applied Mutation:** llm_rewrite
- **Mutation Details:** LLM autonomous rewrite

**Match Type:** ecu_based

**Location:** Lines 2-7

**Injected Content:**

(1)control objective
When this control receives each remote request and reservation request from DCM, it judges whether the remote
air conditioning and reservation air conditioning can be operated.
(2)control overview
Since the function to control air conditioning is divided into ZEV system and ECGW, the arbitration of whether air
conditioning operation of ZEV system is possible and whether air conditioning operation of ECGW is possible is
carried out.
(3)Input data
Table 3-2-1. CAN reception data list
CAN ID
3B6
3FA
the specification omits the dependency between the steps "ECU C accept the request with positive response to ECU B" and "ECU B response with positive response to ECU A"

**Original Context:**
(1)control objective
When this control receives each remote request and reservation request from DCM, it judges whether the remote
air conditioning and reservation air conditioning can be operated.
(2)control overview
Since the function to control air conditioning is divided into ZEV system and ECGW, the arbitration of whether air
conditioning operation of ZEV system is possible and whether air conditioning operation of ECGW is possible is
carried out.
(3)Input data
Table 3-2-1. CAN reception data list
CAN ID
3B6
3FA

---

