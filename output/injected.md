# EARS Rule Injection Results

## Summary

Total matches found: 3
Rules injected: 3
Rules already exist: 0
Rules skipped due to limit: 0

## Air-Conditioning Gateway Function_Ver_1_6.txt

### Rule 13: 3-1. gateway function control

**Status:** inject

**Match Score:** 0.780

**Match Type:** ecu_based

**ECU Match Score:** 1.000

**Condition Match Score:** 0.450

**Location:** Lines 19-20

**Injected Content:**

In this chapter, the air-conditioning gateway control is explained in each section.  
(1) control objective  
  ・Based on the operation information in IDS and remote function, the drive of ventilated seat ECU and steering heater ECU is controlled.  
  ・The actual operation state and failure state are received from the ventilated seat ECU and the steering heater ECU, and the display is instructed to the IDS (the customer is notified).  
  If the ventilated seat ECU does not strictly follow the exact sequence "1) ventilated seat ECU send request to stop other communication to steering heater ECU; 2) ventilated seat ECU send request to start process to steering heater ECU; 3) ventilated seat ECU send request to check the status of process to steering heater ECU; 4) if status is passed or fail, ventilated seat ECU send status report to server", the system behaves accordingly.

**Original Context:**
In this chapter, the air-conditioning gateway control is explained in each section.
(1)control objective
  ・Based on the operation information in IDS and remote function, the drive of ventilated seat ECU and steering
heater ECU is controlled.
  ・The actual operation state and failure state are received from the ventilated seat ECU and the steering heater
ECU, and the display is instructed to the IDS (the customer is notified).

**ECU Context:**
- ECU Line: 19
- ECU Text: ECGW
- Conditions Found: receiving LIN signals from ventilated seat ECU and steering heater ECU, conversion of CAN to LIN signals, conversion of LIN to CAN signals

---

### Rule 24: 3-1-2. Gateway function (conversion from LIN communication to CAN communication)

**Status:** inject

**Match Score:** 0.780

**Match Type:** ecu_based

**ECU Match Score:** 1.000

**Condition Match Score:** 0.450

**Location:** Lines 1-2

**Injected Content:**

(6)Control functions (algorithms)  
  Convert the LIN signals received from each ventilated seat ECU and steering heater ECU to CAN signals and transmit them to the IDS by the method shown in Table 3-1-19~3-1-26, if the sequence "1) Send request 1 from ventilated seat ECU to ECGW; 2) ECGW forward the request 1 to steering heater ECU; 3) steering heater ECU accept the request with positive response to ECGW; 4) ECGW response with positive response to ventilated seat ECU" is not strictly followed.

**Original Context:**
(6)Control functions (algorithms)
  Convert the LIN signals received from each ECU to CAN signals and transmit them to the IDS by the method
shown in Table 3-1 19~3-1-26.

**ECU Context:**
- ECU Line: 1
- ECU Text: ventilated seat ECU
- Conditions Found: operation state received, failure state received

---

### Rule 24: 3-1-1. Gateway/arbitration function (CAN communication →LIN communication conversion)

**Status:** inject

**Match Score:** 0.720

**Match Type:** ecu_based

**ECU Match Score:** 1.000

**Condition Match Score:** 0.300

**Location:** Lines 2-3

**Injected Content:**

■ventilated seat
  The CAN signal received from the IDS is converted into the LIN signal and transmitted to the ventilated sheet
ECU. And, the ventilated sheet is stopped, when the injection and drainage mode notice is received from IDS, to
prevent the battery rising. The LIN signal is transmitted to the ventilated seat ECU according to the remote
operation state received from the DCM.
  Since the switch operation in the IDS is trigger information, the corresponding signal is output for the specified
time only when the operation changes. Since continuous signal transmission is required for driving the ECU, the
signal shall be output according to Table 3-1-6, 3-1-7, Figure 3-1-3, Figure 3-1-4.
  If the preconditions and judgment conditions are not satisfied in each state, the previous value shall be output
without performing state transition.
the sequence "1) Send request 1 from The LIN signal is transmitted to the ventilated seat ECU to Since continuous signal transmission is required for driving the ECU; 2) Since continuous signal transmission is required for driving the ECU forward the request 1 to ECU; 3) ECU accept the request with positive response to Since continuous signal transmission is required for driving the ECU; 4) Since continuous signal transmission is required for driving the ECU response with positive response to The LIN signal is transmitted to the ventilated seat ECU" is not strictly followed

**Original Context:**
■ventilated seat
  The CAN signal received from the IDS is converted into the LIN signal and transmitted to the ventilated sheet
ECU. And, the ventilated sheet is stopped, when the injection and drainage mode notice is received from IDS, to
prevent the battery rising. The LIN signal is transmitted to the ventilated seat ECU according to the remote
operation state received from the DCM.
  Since the switch operation in the IDS is trigger information, the corresponding signal is output for the specified
time only when the operation changes. Since continuous signal transmission is required for driving the ECU, the
signal shall be output according to Table 3-1-6, 3-1-7, Figure 3-1-3, Figure 3-1-4.
  If the preconditions and judgment conditions are not satisfied in each state, the previous value shall be output
without performing state transition.

**ECU Context:**
- ECU Line: 2
- ECU Text: Air-Conditioning Gateway
- Conditions Found: CAN signal from IDS, LIN signal conversion, arbitration of operation information and remote operation

---

