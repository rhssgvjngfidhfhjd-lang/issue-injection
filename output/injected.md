# EARS Rule Injection Results

## Summary

Total matches found: 2
Rules injected: 2
Rules already exist: 0
Rules skipped due to limit: 0

## Sample_ECU_Function_Specification.txt

### Rule 1: 3-1-2. Gateway function (LIN communication to CAN communication conversion)

**Status:** inject

**Match Score:** 0.780

**Match Type:** ecu_based

**ECU Match Score:** 1.000

**Condition Match Score:** 0.450

**Location:** Lines 57-88

**Injected Content:**

(4) Output data  
The converted CAN messages contain:  
- Subsystem status information  
- Diagnostic data  
- Error codes if applicable  

The conversion process occurs after the A/C ECU waits for less than a certain time before sending the next request to the ventilated seat ECU.

**Original Context:**
(4) Output data
The converted CAN messages contain:
- Subsystem status information
- Diagnostic data
- Error codes if applicable

**ECU Context:**
- ECU Line: None
- ECU Text: Climate control ECU

---

### Rule 1: 3-1-1. Gateway/arbitration function (CAN communication to LIN communication conversion)

**Status:** inject

**Match Score:** 0.720

**Match Type:** ecu_based

**ECU Match Score:** 1.000

**Condition Match Score:** 0.300

**Location:** Lines 31-56

**Injected Content:**

(4) Output data  
The gateway ECU sends LIN messages containing:  
- Processed vehicle data  
- System status information  
- Control commands  

The gateway ECU waits for less than a certain time before sending the next request to the A/C ECU.

**Original Context:**
(4) Output data
The gateway ECU sends LIN messages containing:
- Processed vehicle data
- System status information
- Control commands

**ECU Context:**
- ECU Line: None
- ECU Text: gateway ECU
- Conditions Found: receives CAN message, validates message format, checks message ID, converts data format from CAN to LIN, sends LIN message to target ECU

---

