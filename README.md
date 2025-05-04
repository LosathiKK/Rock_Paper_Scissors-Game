Rock-Paper-Scissors Computer Vision Game
This is a computer vision-based Rock-Paper-Scissors game where you can play against your computer using hand gestures captured by your webcam.

Features
Detect hand gestures using your webcam
Play classic Rock-Paper-Scissors
Extended mode: Rock-Paper-Scissors-Lizard-Spock
View image processing steps in real-time
Track game statistics
Take screenshots of gameplay
Folder Structure
rock-paper-scissors/
├── rps.py               # Main game script
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── images/              # Directory for gesture images
    ├── rock.png
    ├── paper.png
    ├── scissors.png
    ├── lizard.png       # For extended mode
    └── spock.png        # For extended mode
Requirements
Python 3.8 or higher
Webcam/front camera
The packages listed in requirements.txt
Installation
Clone or download this repository
Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required packages:
pip install -r requirements.txt
Create an "images" folder and add images for each gesture (optional for visual feedback)
How to Run
Simply run the main script:

python rps.py
How to Play
Press SPACE to start a game
When the countdown begins, say "Rock, Paper, Scissors, Shoot!" out loud
Show your hand gesture when the countdown reaches 0
The computer will randomly select a gesture
The winner will be determined according to the rules
Press 'e' to toggle extended mode (Rock-Paper-Scissors-Lizard-Spock)
Press 's' to take a screenshot
Press ESC to quit
Gesture Guide
Rock: Closed fist
Paper: Open hand with all fingers extended
Scissors: Index and middle fingers extended, forming a V
Lizard (Extended mode): Thumb and pinky extended
Spock (Extended mode): Thumb and index finger extended
Image Processing Visualization
The application displays various image processing steps:

Original frame with hand landmarks
Grayscale conversion
Binary threshold
Edge detection
For Group Projects
This application can be extended and collaborated on by a team of 5-6 members, with each focusing on different aspects:

Computer Vision Expert - Enhance gesture recognition algorithms
UI/UX Designer - Improve the game interface and visual elements
Game Logic Developer - Add game modes and features
Testing and Documentation - Ensure robust functionality and complete documentation
Performance Optimizer - Make the code run efficiently
Extensions Developer - Add advanced features like multiplayer or gesture customization
Troubleshooting
If the webcam doesn't open, check if another application is using it
If gesture detection is poor, try adjusting lighting conditions
If the application runs slowly, close other resource-intensive applications
"# Rock_Paper_Scissors-Game" 
