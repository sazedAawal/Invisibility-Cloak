import cv2
import numpy as np
import mediapipe as mp

LEARN_RATE = 0.05 # how fast clean (non-person) pixels get folded into the background
MASK_BLUR  = 9 # softens the mask edges so the live invisibility blend looks natural
MENU_HEIGHT = 40 # height of the control menu at the bottom

class BackgroundLearner:
    """Keeps a running background image, updated only where there is no person."""

    def __init__(self):
        self.bg = None

    def update(self, frame, person_mask_bin):
        frame_f = frame.astype(np.float32)
        if self.bg is None:
            self.bg = frame_f.copy()  # seed it; will self-correct as you move
            return
        background_pixels = ~person_mask_bin
        self.bg[background_pixels] = (
            self.bg[background_pixels] * (1 - LEARN_RATE)
            + frame_f[background_pixels] * LEARN_RATE
        )

    def reset(self):
        self.bg = None


def get_person_mask(segmenter, frame):
    """Ask MediaPipe 'which pixels are a person?' -> soft mask, values 0 (bg) to 1 (person)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)
    return np.clip(result.segmentation_mask, 0, 1)

def draw_bottom_menu(frame, is_invisible):
    """Appends a dark menu bar to the bottom of the frame with shortcut guides."""
    # Create a black bar matching the width of the frame
    h, w, c = frame.shape
    menu_bar = np.zeros((MENU_HEIGHT, w, c), dtype=np.uint8)
    
    # Define text strings
    status_text = "STATUS: INVISIBLE" if is_invisible else "STATUS: VISIBLE"
    status_color = (0, 255, 255) if is_invisible else (0, 255, 0)
    controls_text = "[SPACE]: Toggle Cloak  |  [R]: Reset  |  [Q]: Quit"
    
    # Draw Status (Left-aligned)
    cv2.putText(menu_bar, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
    
    # Draw Controls (Right-aligned calculation)
    text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    text_x = w - text_size[0] - 15
    cv2.putText(menu_bar, controls_text, (text_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Stack the video frame and the menu bar vertically
    return np.vstack((frame, menu_bar))

def main():
    # 1. INITIALIZE WEB FEED
    # Open a connection to the default system camera (0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Close any other app using it (Zoom/Teams) and try again.")
        return
    
    # 2. INITIALIZE MEDIAPIPE & AUXILIARY CLASS OBJECTS
    # model_selection=1 optimizes the segmentation model for a full/upper body shot 
    # instead of a close-up portrait (0).
    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    bg_learner = BackgroundLearner()
    window = "Invisibility Cloak"

    print("Move around a little so the background can learn what's behind you.")
    print("SPACE = vanish/appear, R = relearn background, Q = quit.")
    
    # Global state variable tracking whether the cloak is activated
    invisible = False

    # 3. CORE PROCESSING LOOP
    while True:
        # Capture frame-by-frame from the camera feed
        ret, frame = cap.read()
        if not ret:
            continue # Skip processing if a frame wasn't successfully grabbed
        
        # 4. SEGMENTATION & BACKGROUND LEARNING
        # Extract a soft probability mask where 1.0 = human, 0.0 = background
        soft_mask = get_person_mask(segmenter, frame)
        
        # Create a hard boolean mask (True/False) at a 50% confidence threshold.
        # This tells the BackgroundLearner precisely which pixels are safe to learn from.
        person_bin = soft_mask > 0.5
        bg_learner.update(frame, person_bin)
        
        # 5. VISUAL RENDERING LOGIC (CLOAK ACTIVATED vs. DEACTIVATED)
        if invisible and bg_learner.bg is not None:
            # Smooth out the edges of the soft mask using Gaussian Blur.
            # This softens the transition boundary between the player and the backdrop.
            blurred_mask = cv2.GaussianBlur(soft_mask, (MASK_BLUR, MASK_BLUR), 0)
            # Expand mask dimensions from 2D (H, W) to 3D (H, W, 1) so it broadcasts 
            # seamlessly across the 3-channel (BGR) frame arrays during multiplication.
            mask_3ch = blurred_mask[:, :, np.newaxis]
            # Linear Interpolation (Alpha Blending) Formula:
            # Result = (Live Frame * Inverse Mask) + (Learned Background * Mask)
            # - Where mask is 1 (human), it multiplies by 0 on the live frame side, 
            #   and multiplies by 1 on the background side -> effectively replacing you.
            blended = frame.astype(np.float32) * (1 - mask_3ch) + bg_learner.bg * mask_3ch
            # Clip values to ensure they stay within standard 8-bit color depth bounds [0-255]
            display = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            # If cloak is off (or background isn't learned yet), fallback to normal feed
            display = frame.copy()
        
        # 6. UI COMPOSITION & DISPLAY
        # Generate the UI canvas by appending the interactive menu bar at the bottom
        ui_display = draw_bottom_menu(display, invisible)
        cv2.imshow(window, ui_display)

        # 7. KEYBOARD EVENT LISTENER
        # Wait 1 millisecond for user key input; mask with 0xFF for standard ASCII translation
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            invisible = not invisible
        elif key == ord('r'):
            bg_learner.reset()
            print("Background reset - move around again to relearn it.")
    
    # 8. RESOURCE CLEANUP
    # Release the video hardware resource and destroy generated windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()