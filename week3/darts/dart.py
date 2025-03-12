import cv2, pyautogui, keyboard, time, numpy as np
cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

template = cv2.imread('template.png', 0)
if template is None:
    print("Template didn't load!")
    exit()
print("Starting Dart game! Press 'q' to end the game.")

while True:
    if keyboard.is_pressed("q"):
        print("Ending Dart game!")
        break

    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

    result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f"Detection confidence: {max_val:.3f}")

    template_height, template_width = template.shape
    top_left_x, top_left_y  = max(0, max_loc[0] - 10), max(0, max_loc[1] - 10)
    bottom_right_x, bottom_right_y = min(result.shape[1], max_loc[0] + template_width + 10), min(result.shape[0], max_loc[1] + template_height + 10)
    roi = result[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    X, Y = np.meshgrid(np.arange(top_left_x, bottom_right_x), np.arange(top_left_y, bottom_right_y))
    mask = roi > 0.5
    
    if np.sum(mask) > 0:
        weighted_x, weighted_y  = np.average(X[mask], weights=roi[mask]), np.average(Y[mask], weights=roi[mask])
        center_x, center_y  = int(weighted_x + template_width / 2), int(weighted_y + template_height / 2)
    else: center_x, center_y = max_loc[0] + template_width // 2, max_loc[1] + template_height // 2

    if max_val > 0.5:
        cv2.rectangle(screenshot_np, max_loc, (max_loc[0] + template_width, max_loc[1] + template_height), (0, 255, 0), 2)
        print(f"Target locked at (x: {center_x}, y: {center_y}) with confidence: {max_val:.3f}!")
        pyautogui.moveTo(center_x, center_y)
        pyautogui.click()
        time.sleep(0.3)

    cv2.imshow("Debug", screenshot_np)
    if cv2.waitKey(1) & 0xFF == 27: break
cv2.destroyAllWindows()