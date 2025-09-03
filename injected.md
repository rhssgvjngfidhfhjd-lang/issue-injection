# EARS Rule Injection Results

## Summary

Total matches found: 1
Rules injected: 1
Rules already exist: 0
Rules skipped due to limit: 0

## Air-Conditioning Gateway Function_Ver_1_6.txt

### Rule 3: 3-1-2. Gateway function (conversion from LIN communication to CAN communication)

**Status:** inject

**Match Score:** 1.000

**Match Type:** fallback

**Location:** Lines 1532-1939

**Injected Content:**

(3)Input data
  See "SUZUKI LIN SPECIFICATION" for details of LIN data.
  Table 3-1-16. LIN received data list
      LIN ID     LIN signal name                                               Data Summary
                                                                               Ventilated seat operating information
                 Front Seat Vent Mode(Driver)                                  received from the ventilated seat ECU
                                                                               (driver's side)
                                                                               Ventilated seat operation information received
                 Front Seat Vent Mode(Passenger)
                                                                               from the ventilated seat ECU (passenger side)
                                                                               Ventilated seat failure information received
                 Front Seat Vent Blower Status(Driver)
                                                                               from the ventilated seat ECU (driver's side)
                                                                               Ventilated seat failure information received
                 Front Seat Vent Blower Status(Passenger)                      from the ventilated seat ECU (passenger
                                                                               side)
                                                                               Seat heater operation information received
                 Front Seat Heater Mode(Driver)
                                                                               from the ventilated seat ECU (driver's side)
                                                                               Seat heater operation information received
                 Front Seat Heater Mode(Passenger)
                                                                               from ventilated seat ECU (passenger side)
                                                                               Seat heater failure information received from
      2D         Front Seat Heater Status(Driver)
                                                                               the ventilated seat ECU (driver's side)
                                                                               Seat heater failure information received from
                 Front Seat Heater Status(Passenger)
                                                                               the ventilated seat ECU (passenger side)
                                                                               LIN communication error received from
                 LIN Communication Error(Front Seat)
                                                                               ventilated seat ECU
                                                                               Ventilated seat equipment information
                 Front Seat Vent Blower Equipped (Driver)                      received from the ventilated seat ECU
                                                                               (driver's side)
                                                                               Ventilated seat equipment information
                 Front Seat Vent Blower Equipped
                                                                               received from the ventilated seat ECU
                 (Passenger)
                                                                               (passenger side)
                                                                               Seat heater installation information received
                 Front Seat Heater Equipped(Driver)
                                                                               from ventilated seat ECU (driver's side)
                                                                               Seat heater installation information received
                 Front Seat Heater Equipped (Passenger)
                                                                               from ventilated seat ECU (passenger side)
                                                                               Operating information of the steering heater
                 Steering Heater Mode
                                                                               received from the steering heater ECU
                                                                               Steering heater failure information received
                 Steering Heater Status
                                                                               from the steering heater ECU
      29
                                                                               LIN communication error received from
                 LIN Communication Error(Steering Heater)
                                                                               steering heater ECU
                                                                               Steering heater installation information
                 Steering Heater Equipped(LIN)
                                                                               received from the steering heater ECU
received from the ventilated seat ECU sends a request to from the ventilated seat ECU start a process without previously sending a request to from the ventilated seat ECU to stop other communications,

**Original Context:**
(3)Input data
  See "SUZUKI LIN SPECIFICATION" for details of LIN data.
  Table 3-1-16. LIN received data list
      LIN ID     LIN signal name                                               Data Summa...

---

