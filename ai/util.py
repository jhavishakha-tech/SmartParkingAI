import cv2

# =====================================================
# SLOT SIZE
# Change these if you want bigger/smaller rectangles
# =====================================================
SLOT_WIDTH = 20
SLOT_HEIGHT = 42

# Overlap threshold
OVERLAP_THRESHOLD = 0.10


def intersection_area(ax1, ay1, ax2, ay2,
                      bx1, by1, bx2, by2):

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0

    return (x_right - x_left) * (y_bottom - y_top)


def checkParkingSpace(
    img,
    posList,
    car_boxes,
    w=SLOT_WIDTH,
    h=SLOT_HEIGHT,
    overlap_thresh=OVERLAP_THRESHOLD,
):

    free = 0

    slot_area = w * h

    for pos in posList:

        x, y = pos

        sx1 = x
        sy1 = y
        sx2 = x + w
        sy2 = y + h

        occupied = False

        for (cx1, cy1, cx2, cy2) in car_boxes:

            # --------------------------
            # Car Center
            # --------------------------
            center_x = (cx1 + cx2) // 2
            center_y = (cy1 + cy2) // 2

            if (
                sx1 <= center_x <= sx2
                and
                sy1 <= center_y <= sy2
            ):
                occupied = True
                break

            # --------------------------
            # Area Overlap
            # --------------------------
            inter = intersection_area(
                sx1,
                sy1,
                sx2,
                sy2,
                cx1,
                cy1,
                cx2,
                cy2,
            )

            overlap = inter / slot_area

            if overlap >= overlap_thresh:
                occupied = True
                break

        if occupied:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
            free += 1

        cv2.rectangle(
            img,
            (sx1, sy1),
            (sx2, sy2),
            color,
            2,
        )

    return free