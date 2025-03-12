import cv2, os, pyautogui, keyboard, time, numpy as np
cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

template_folder = 'ducks_template'
template_paths = [os.path.join(template_folder, f) for f in os.listdir(template_folder) if f.lower().endswith('.png')]
templates = [cv2.imread(path, 0) for path in template_paths]

if not templates:
    print("No duck templates found in folder 'ducks'!")
    exit()
print("Starting DuckHunt multi-scale! Press 'q' to end the game.")

while True:
    if keyboard.is_pressed("q"):
        print("Ending DuckHunt!")
        break

    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

    duck_found_this_frame = False
    for template in templates:
        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.5:
            template_height, template_width = template.shape

            top_left_x, top_left_y = max(0, max_loc[0] - 10), max(0, max_loc[1] - 10)
            bottom_right_x, bottom_right_y = min(result.shape[1], max_loc[0] + template_width + 10), min(result.shape[0], max_loc[1] + template_height + 10)
            roi = result[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            X, Y = np.meshgrid(np.arange(top_left_x, bottom_right_x), np.arange(top_left_y, bottom_right_y))
            mask = roi > 0.5

            if np.sum(mask) > 0:
                weighted_x, weighted_y = np.average(X[mask], weights=roi[mask]), np.average(Y[mask], weights=roi[mask])
                center_x, center_y = int(weighted_x + template_width / 2), int(weighted_y + template_height / 2)
            else:
                center_x, center_y = max_loc[0] + template_width // 2, max_loc[1] + template_height // 2

            cv2.rectangle(screenshot_np, max_loc, (max_loc[0] + template_width, max_loc[1] + template_height), (0, 255, 0), 2)
            print(f"Target locked at (x: {center_x}, y: {center_y}) with confidence: {max_val:.3f}!")
            pyautogui.moveTo(center_x, center_y)
            pyautogui.click()
            time.sleep(0.3)
            duck_found_this_frame = True
            break

    cv2.imshow("Debug", screenshot_np)
    if cv2.waitKey(1) & 0xFF == 27: break
cv2.destroyAllWindows()