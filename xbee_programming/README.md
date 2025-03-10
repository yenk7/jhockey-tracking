# For Course TAs: XBee Setup and Deployment Instructions

Author: Anway Pimpalkar | Last Updated: March 10th 2025

## Starting the Game

1. Switch on the Intel NUC, make sure the camera and Tx module are connected.
2. In a terminal/cmd shell, enter `./start.sh`.

## Xbee Module Setup

### Requirements

1. XBee 3 Module.
2. SparkFun XBee Explorer USB.
3. Machine running Linux/Windows/MacOS with [Digi XCTU](https://hub.digi.com/support/products/xctu/) installed.

### First-Time Setup

1. Open XCTU.
2. Click on the Discover button in the top left corner.
3. Select all ports to be scanned, and press next.
4. Select bauds of `9600` and `115200`, leave the rest at default. Press finish.
5. Select and add your module.
    - For new modules, you'll see a R in the bottom left of the icon.
    - For previously setup modules, you'll see an E at the bottom left.
6. Click on the module from the list of radio modules.
7. Either load an existing profile, or manually set the parameters.

#### Option 1: Load Existing Profile

1. Click on Profile, select apply configuration profile.
2. Open the `/xbee_config/xbee_rx_config.xpro` configuration profile.
3. Update the existing firmware and apply the settings.
4. In Discovery Options, change the following setting:
    - `Node Identifier: [Insert ROBOT_ID]` (A/B/C..)
5. Click on Write to save.

**Note: For the Tx module, load the `/xbee_config/xbee_tx_config.xpro` module instead. No MicroPython script is necessary for it.**

#### Option 2: Manually Set Settings

1. Click on update.
2. Select `XB3-24`, `Digi XBee3 802.15.4 TH`, and `200D (Newest)`. Proceed to update.
3. Once updated, there will be a default set of parameters already loaded onto the board. Change the following set of properties:
   - Networking:
     - `Channel: C`
     - `Network PAN ID: 2024`
  
   - Discovery Options: 
     - `Node Identifier: [Insert ROBOT_ID]` (XY: X is the team ID, Y is either A or B)

   - Coordinator/End Device Configuration:
     - `Device Role: End Device[0]`
  
   - Addressing:
     - `16-bit Source Address: 2`
     - `Destination Address High: 0`
     - `Destination Address Low: 1`
  
   - UART Interface:
     - `UART Baud Rate: 115200`

4. Click on write to save the changes.

## XBee MicroPython Setup

### Requirements

1. XBee 3 Module.
2. SparkFun XBee Explorer USB.
3. Machine running Windows 10. <span style="color:red">(Ran into multiple issues with MacOS, have not tested functionality with Linux)</span>
4. Installation of [PyCharm Community Edition <span style="color:red">2023.2.5</span>](https://www.jetbrains.com/pycharm/download/other.html). Latest versions will not work.

### Installing Digi XBee MicroPython PyCharm IDE Plugin

1. Open PyCharm.
2. Go to the Plugins window by doing one of the following:
3. Select Configure > Plugins if you are on the Welcome screen, or
4. Select File > Settings > Plugins if you have a project open.
5. Type `Digi XBee` in the Marketplace search box.
6. Click Install.
7. When finished, click Restart IDE to complete the plugin installation.

### Deploying MicroPython Script

1. Open the `xbee_rx` PyCharm project.
2. Replace the `ROBOT_ID` variable in the MicroPython script with the unique identifier assigned to the XBee module.
3. Select the appropriate port connected the device is connected to, and upload.