import cv2
import numpy as np
import mediapipe as mp
import random
import time
import os
from datetime import datetime

from enum import Enum

# Define game states as an enum
class GameState(Enum):
    WAITING = 0     # Waiting for player to start game
    COUNTDOWN = 1   # Countdown before capturing gesture
    CAPTURE = 2     # Capturing the player's gesture
    RESULT = 3      # Displaying the result

class RPSGame:
    def __init__(self):
        # Initialize game state
        self.state = GameState.WAITING
        self.player_move = None
        self.computer_move = None
        self.result = None
        self.countdown_end = 0
        self.result_display_end = 0
        self.player_score = 0
        self.computer_score = 0
        self.extended_mode = False
        self.last_screenshot_time = 0
    
        # Initialize hand landmarks attribute
        self.hand_landmarks = None
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Game variables
        self.choices = ["rock", "paper", "scissors"]
        self.extended_choices = ["rock", "paper", "scissors", "lizard", "spock"]
        self.user_choice = None
        self.computer_choice = None
        self.result = None
        self.game_active = False
        self.countdown_started = False
        self.countdown_time = 0
        self.result_time = 0
        self.show_result = False
        self.extended_mode = False  # Set to True to play Rock, Paper, Scissors, Lizard, Spock
        
        # Load gesture images
        self.gesture_images = {}
        for gesture in self.extended_choices:
            path = f"images/{gesture}.png"
            if os.path.exists(path):
                self.gesture_images[gesture] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            else:
                print(f"Warning: Could not find image for {gesture} at {path}")
        
        # Create directory for images if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')
            print("Created 'images' directory. Please add gesture images there.")
        
        # Win/loss statistics
        self.stats = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
        
        # Processing display flags
        self.show_binary = False
        self.show_grayscale = False
        self.show_edges = False
        self.show_contours = False
        
    def create_processing_visualization(self, frame):
        h, w = frame.shape[:2]
        # Create a 2x2 grid for visualization
        grid = np.zeros((h, w*2, 3), dtype=np.uint8)
    
        # Original frame in top-left
        grid[:h//2, :w] = frame[:h//2]
    
        # Convert to grayscale for processing visualization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR for display
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
        # Resize gray_bgr to match target dimensions before assignment
        gray_bgr_resized = cv2.resize(gray_bgr, (w, h//2))
    
        # Processed frame in top-right
        grid[:h//2, w:w*2] = gray_bgr_resized
    
        # Hand landmarks visualization in bottom-left
        hand_vis = frame.copy()
        if self.hand_landmarks:
            self.mp_drawing.draw_landmarks(
                hand_vis, 
                self.hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS)
        grid[h//2:, :w] = hand_vis[h//2:]
    
        # Game state in bottom-right
        info_display = np.zeros((h//2, w, 3), dtype=np.uint8)
    
        # Show countdown or result
        if self.state == GameState.COUNTDOWN:
            time_left = int(self.countdown_end - time.time()) + 1
            if time_left > 0:
                cv2.putText(info_display, str(time_left), (w//2-50, h//4+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
        elif self.state == GameState.RESULT:
            # Show player's move
            player_move_text = f"Your move: {self.player_move}"
            cv2.putText(info_display, player_move_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show computer's move
            computer_move_text = f"Computer: {self.computer_move}"
            cv2.putText(info_display, computer_move_text, (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show game result
            result_color = (0, 255, 0) if self.result == "You win!" else \
                        (0, 0, 255) if self.result == "Computer wins!" else (255, 255, 255)
            cv2.putText(info_display, self.result, (20, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, result_color, 2)
            
            # Show score
            score_text = f"Score: You {self.player_score} - {self.computer_score} Computer"
            cv2.putText(info_display, score_text, (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show mode
        mode_text = "Mode: " + ("Standard" if not self.extended_mode else "Extended (Rock-Paper-Scissors-Lizard-Spock)")
        cv2.putText(info_display, mode_text, (20, h//4-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add the status display to bottom-right
        grid[h//2:, w:w*2] = info_display
        
        return grid
    
    def detect_gesture(self, hand_landmarks):
        """Detect the gesture based on MediaPipe hand landmarks"""
        # Get tip and pip landmarks for each finger
        finger_tips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        finger_pips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        ]
        
        # Check if fingers are extended
        fingers_extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            if tip.y < pip.y:  # If tip is above pip (y increases downward)
                fingers_extended.append(True)
            else:
                fingers_extended.append(False)
        
        # Special case for thumb - check if thumb tip is to the right/left of the thumb ip
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = finger_tips[0]
        thumb_ip = finger_pips[0]
        
        # Determine if the hand is left or right
        if wrist.x < thumb_tip.x:  # Right hand
            fingers_extended[0] = thumb_tip.x > thumb_ip.x
        else:  # Left hand
            fingers_extended[0] = thumb_tip.x < thumb_ip.x
        
        # Detect basic gestures
        if not any(fingers_extended):
            return "rock"
        elif all(fingers_extended):
            return "paper"
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not fingers_extended[3] and not fingers_extended[4]:
            return "scissors"
        elif self.extended_mode:
            # For extended mode (Rock, Paper, Scissors, Lizard, Spock)
            if fingers_extended[0] and fingers_extended[4] and not fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3]:
                return "spock"
            elif fingers_extended[0] and fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
                return "lizard"
        
        return None
    
    def determine_winner(self, user_choice, computer_choice):
        """Determine the winner based on Rock-Paper-Scissors(-Lizard-Spock) rules"""
        if user_choice == computer_choice:
            return "Tie!"
        
        # Rules for standard Rock-Paper-Scissors
        if not self.extended_mode:
            if (user_choice == "rock" and computer_choice == "scissors") or \
               (user_choice == "paper" and computer_choice == "rock") or \
               (user_choice == "scissors" and computer_choice == "paper"):
                return "You win!"
            else:
                return "Computer wins!"
        else:
            # Rules for Rock-Paper-Scissors-Lizard-Spock
            # Scissors cuts Paper, Paper covers Rock, Rock crushes Lizard, 
            # Lizard poisons Spock, Spock smashes Scissors, Scissors decapitates Lizard, 
            # Lizard eats Paper, Paper disproves Spock, Spock vaporizes Rock, Rock crushes Scissors
            win_conditions = {
                "scissors": ["paper", "lizard"],
                "paper": ["rock", "spock"],
                "rock": ["lizard", "scissors"],
                "lizard": ["spock", "paper"],
                "spock": ["scissors", "rock"]
            }
            
            if computer_choice in win_conditions.get(user_choice, []):
                return "You win!"
            else:
                return "Computer wins!"
    
    def overlay_computer_choice(self, frame, choice):
        """Overlay the computer's choice on the frame"""
        if choice in self.gesture_images:
            img = self.gesture_images[choice]
            if img is not None:
                h, w = frame.shape[:2]
                img_h, img_w = img.shape[:2]
                
                # Resize image to fit nicely on screen
                scale = min(h/3 / img_h, w/3 / img_w)
                new_h, new_w = int(img_h * scale), int(img_w * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
                
                # Calculate position (top right corner)
                x_offset = w - new_w - 10
                y_offset = 10
                
                # If image has alpha channel (transparent background)
                if img_resized.shape[2] == 4:
                    alpha = img_resized[:, :, 3] / 255.0
                    for c in range(3):
                        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = \
                            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] * (1 - alpha) + \
                            img_resized[:, :, c] * alpha
                else:
                    # Just copy the image over
                    frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        # Add text label
        cv2.putText(frame, f"Computer: {choice.upper()}", 
                    (frame.shape[1] - 250, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """Run the Rock-Paper-Scissors game"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Toggle for image processing visualization
        process_viz_key = 'p'  # Press 'p' to toggle processing visualization
        
        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
                
            # Flip the frame horizontally for a selfie-view
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image to detect hands
            results = self.hands.process(rgb_frame)
            self.hand_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
            
            # Toggle processing visualizations
            self.show_grayscale = True
            self.show_binary = True
            self.show_edges = True
            
            # Create visualization grid
            display = self.create_processing_visualization(frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display[:frame.shape[0], :frame.shape[1]],
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect gesture
                    gesture = self.detect_gesture(hand_landmarks)
                    if gesture:
                        cv2.putText(display, f"Detected: {gesture.upper()}", 
                                   (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Game logic
            current_time = time.time()
            
            # Display instructions and stats
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, h-100), (w, h), (0, 0, 0), -1)
            cv2.putText(display, f"Press 'SPACE' to play | 'ESC' to quit | 'e' for extended mode: {'ON' if self.extended_mode else 'OFF'}", 
                       (10, h-70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Stats: Wins: {self.stats['wins']} | Losses: {self.stats['losses']} | Ties: {self.stats['ties']} | Total: {self.stats['total']}", 
                       (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Handle countdown and game state
            if self.game_active:
                if not self.countdown_started:
                    self.countdown_started = True
                    self.countdown_time = current_time + 3  # 3 second countdown
                
                if current_time < self.countdown_time:
                    remaining = int(self.countdown_time - current_time) + 1
                    cv2.putText(display, f"Say 'Rock, Paper, Scissors' and Show Your Hand: {remaining}", 
                              (w//4, h//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if not self.show_result:
                        # Capture user's gesture
                        if results.multi_hand_landmarks:
                            self.user_choice = self.detect_gesture(results.multi_hand_landmarks[0])
                        else:
                            self.user_choice = None
                        
                        # Computer makes a choice
                        if self.extended_mode:
                            self.computer_choice = random.choice(self.extended_choices)
                        else:
                            self.computer_choice = random.choice(self.choices)
                        
                        # Determine winner
                        if self.user_choice:
                            self.result = self.determine_winner(self.user_choice, self.computer_choice)
                            
                            # Update stats
                            self.stats["total"] += 1
                            if "win" in self.result.lower():
                                self.stats["wins"] += 1
                            elif "tie" in self.result.lower():
                                self.stats["ties"] += 1
                            else:
                                self.stats["losses"] += 1
                        else:
                            self.result = "No gesture detected!"
                        
                        self.show_result = True
                        self.result_time = current_time + 3  # Show result for 3 seconds
                    
                    # Display result
                    if current_time < self.result_time:
                        # Overlay computer's choice
                        if self.computer_choice:
                            display = self.overlay_computer_choice(display, self.computer_choice)
                        
                        # Display user's choice and result
                        if self.user_choice:
                            cv2.putText(display, f"You: {self.user_choice.upper()}", 
                                       (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        else:
                            cv2.putText(display, "You: No gesture detected", 
                                       (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        
                        # Display result
                        cv2.putText(display, self.result, 
                                   (w//3, h//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    else:
                        # Reset game state
                        self.game_active = False
                        self.countdown_started = False
                        self.show_result = False
            
            # Display the frame
            cv2.imshow('Rock-Paper-Scissors Game', display)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on ESC
            if key == 27:  # ESC key
                break
            
            # Start game on SPACE
            elif key == 32:  # SPACE key
                if not self.game_active:
                    self.game_active = True
            
            # Toggle extended mode on 'e'
            elif key == ord('e'):
                self.extended_mode = not self.extended_mode
                print(f"Extended mode: {'ON' if self.extended_mode else 'OFF'}")
                
            # Toggle processing visualization on 'p'
            elif key == ord(process_viz_key):
                self.show_grayscale = not self.show_grayscale
                self.show_binary = not self.show_binary
                self.show_edges = not self.show_edges
            
            # Take screenshot on 's'
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"rps_screenshot_{timestamp}.png", display)
                print(f"Screenshot saved as rps_screenshot_{timestamp}.png")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    print("Starting Rock-Paper-Scissors Game...")
    print("Instructions:")
    print("1. Press SPACE to start a game")
    print("2. Show your hand gesture when the countdown ends")
    print("3. Press 'e' to toggle extended mode (Rock-Paper-Scissors-Lizard-Spock)")
    print("4. Press 's' to take a screenshot")
    print("5. Press ESC to quit")
    
    game = RPSGame()
    game.run()