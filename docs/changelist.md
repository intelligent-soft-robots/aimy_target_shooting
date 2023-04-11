# Changelist

| Extension | Status | Comment |
|---|---|---|
| Programmable Wheel <br>Spinning Direction | open | Large spin can be achieved with AIMY by opposite <br>spinning wheels.<br>The current system, however, does only allow <br>spinning in one direction. The reason for <br>uni-directional spinning is the electronic speed <br>controller from T-Motor which AIMY currently uses. <br>In a future version, we recommend equipping AIMY <br>with the controller board F28069M LaunchPad from <br>Texas Instruments and two DRV8305 motor driver.<br>This hardware modification also requires a new <br>API with the Raspberry Pi and a low-level speed <br>control of the three-phase motor currents. |
| Speed Control with <br>Speed Sensors | open | Currently, the speed of the throwing wheels is <br>controlled in an open-loop fashion. For higher <br>precision, each throwing wheel should be equipped <br>with a high-resolution speed/angle sensor. <br>Then a speed controller should regulate the motor <br>currents to provide reliable and robust speed <br>actuation. |