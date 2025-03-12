import cv2, pyautogui, time, numpy as np
from pynput import keyboard as pynput_keyboard
cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

template = cv2.imread('template.png', 0)
if template is None:
    print("Template didn't load!")
    exit()
print("Starting Dart game! Press 'q' to end the game.")

exit_flag = False
def on_press(key):
    global exit_flag
    try:
        if key.char == 'q':
            exit_flag = True
            return False
    except AttributeError: pass
listener = pynput_keyboard.Listener(on_press=on_press)
listener.start()

while True:
    if exit_flag:
        print("Ending Dart game!")
        break

    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

    screen_width, screen_height = pyautogui.size()
    screenshot_height, screenshot_width = screenshot_np.shape[:2]
    scale_x, scale_y = screen_width / screenshot_width, screen_height / screenshot_height

    result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f"Detection confidence: {max_val:.3f}")

    template_height, template_width = template.shape
    top_left_x, top_left_y = max(0, max_loc[0]), max(0, max_loc[1])
    bottom_right_x, bottom_right_y = min(result.shape[1], max_loc[0] + template_width), min(result.shape[0], max_loc[1] + template_height)
    #top_left_x, top_left_y = max(0, max_loc[0] - 10), max(0, max_loc[1] - 10)
    #bottom_right_x, bottom_right_y = min(result.shape[1], max_loc[0] + template_width + 10), min(result.shape[0], max_loc[1] + template_height + 10)
    roi = result[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    X, Y = np.meshgrid(np.arange(top_left_x, bottom_right_x), np.arange(top_left_y, bottom_right_y))
    mask = roi > 0.5
    
    if np.sum(mask) > 0:
        weighted_x, weighted_y = np.average(X[mask], weights=roi[mask]), np.average(Y[mask], weights=roi[mask])
        center_x, center_y = int(weighted_x + template_width / 2), int(weighted_y + template_height / 2)
    else: center_x, center_y = max_loc[0] + template_width // 2, max_loc[1] + template_height // 2

    center_x_scaled, center_y_scaled = int(center_x * scale_x), int(center_y * scale_y)

    if max_val > 0.5:
        cv2.rectangle(screenshot_np, max_loc, (max_loc[0] + template_width, max_loc[1] + template_height), (0, 255, 0), 2)
        print(f"Target locked at (x: {center_x_scaled}, y: {center_y_scaled}) with confidence: {max_val:.3f}!")
        pyautogui.moveTo(center_x_scaled, center_y_scaled)
        pyautogui.click()
        time.sleep(0.1)

    cv2.imshow("Debug", screenshot_np)
    #if cv2.waitKey(1) & 0xFF == 27: break
cv2.destroyAllWindows()