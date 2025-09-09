# EARS Rule Injection Results

## Summary

Total matches found: 3
Rules injected: 3
Rules already exist: 0
Rules skipped due to limit: 0

## Air-Conditioning Gateway Function_Ver_1_6.txt

### Rule 1: 3-1. gateway function control

**Status:** inject

**Match Score:** 0.780

**Match Type:** ecu_based

**ECU Match Score:** 1.000

**Condition Match Score:** 0.450

**Location:** Lines 3-4

**Injected Content:**

In this chapter, the air-conditioning gateway control is explained in each section.  
(1) control objective  
  ・Based on the operation information in IDS and remote function, the drive of ventilated seat ECU and steering heater ECU is controlled.  
  ・The actual operation state and failure state are received from the ventilated seat ECU and the steering heater ECU, and the display is instructed to the IDS (the customer is notified), where the ventilated seat ECU waits for less than a certain time before sending the next request to the steering heater ECU.

**Original Context:**
In this chapter, the air-conditioning gateway control is explained in each section.
(1)control objective
  ・Based on the operation information in IDS and remote function, the drive of ventilated seat ...

**ECU Context:**
- ECU Line: 3
- ECU Text: integrated display (IDS)
- Conditions Found: Based on the operation information in IDS and remote function, The actual operation state and failure state are received

---

### Rule 1: 3-1-1. Gateway/arbitration function (CAN communication →LIN communication conversion)

**Status:** inject

**Match Score:** 0.720

**Match Type:** ecu_based

**ECU Match Score:** 1.000

**Condition Match Score:** 0.300

**Location:** Lines 1-2

**Injected Content:**

■ventilated seat  
The CAN signal received from the IDS is converted into the LIN signal and transmitted to the ventilated seat ECU. And, the ventilated sheet is stopped, when the injection and drainage mode notice is received from IDS, to prevent the battery rising. The LIN signal is transmitted to the ventilated seat ECU according to the remote operation state received from the DCM.  
Since the switch operation in the IDS is trigger information, the corresponding signal is output for the specified time only when the operation changes. Since continuous signal transmission is required for driving the ECU, the signal shall be output according to Table 3-1-6, 3-1-7, Figure 3-1-3, Figure 3-1-4.  
If the DCM waits for less than a certain time before sending the next request to the ventilated seat ECU, the previous value shall be output without performing state transition.

**Original Context:**
■ventilated seat
  The CAN signal received from the IDS is converted into the LIN signal and transmitted to the ventilated sheet
ECU. And, the ventilated sheet is stopped, when the injection and drain...

**ECU Context:**
- ECU Line: 1
- ECU Text: Air-Conditioning Gateway
- Conditions Found: Operation in IDS, Remote operation information, CAN signal by the operation in IDS is converted into LIN signal

---

### Rule 1: 3-1-3. fault diagnosis output function

**Status:** inject

**Match Score:** 0.720

**Match Type:** ecu_based

**ECU Match Score:** 1.000

**Condition Match Score:** 0.300

**Location:** Lines 2-3

**Injected Content:**

Front Seat Heater Status(Driver)

The front seat heater control function, managed by the ventilated seat ECU, requires coordination with the A/C ECU to regulate heating output based on temperature settings and climate control input. The ventilated seat ECU wait for less than a certain time before send next request to A/C ECU to ensure optimal thermal response and prevent rapid successive demands that could degrade system performance. The interaction between the ventilated seat ECU and A/C ECU remains synchronized through periodic status checks and controlled message timing to maintain user comfort and safety.

**Original Context:**
Front Seat Heater Status(Driver)

**ECU Context:**
- ECU Line: 2
- ECU Text: ECGW
- Conditions Found: Receives LIN communication errors, Receives failure information from other ECUs

---

