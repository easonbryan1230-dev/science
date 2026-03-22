import numpy as np
import matplotlib.pyplot as plt

def get_next_billiard_point(A, B, a, b):
    """計算撞球在橢圓內的下一個反射點 (通用於所有焦散線)"""
    nx, ny = B[0] / a**2, B[1] / b**2
    nn = np.hypot(nx, ny)
    nx, ny = nx / nn, ny / nn

    vx, vy = B[0] - A[0], B[1] - A[1]
    vn = np.hypot(vx, vy)
    vx, vy = vx / vn, vy / vn

    dot = vx * nx + vy * ny
    ux = vx - 2 * dot * nx
    uy = vy - 2 * dot * ny

    c2 = (ux / a)**2 + (uy / b)**2
    c1 = 2 * (B[0] * ux / a**2 + B[1] * uy / b**2)
    t = -c1 / c2

    C = np.array([B[0] + t * ux, B[1] + t * uy])
    return C

def calculate_excenters(A, B, C):
    """計算三角形 ABC 的旁心"""
    side_a = np.linalg.norm(B - C)
    side_b = np.linalg.norm(A - C)
    side_c = np.linalg.norm(A - B)

    W_A = -side_a + side_b + side_c
    J_A = (-side_a * A + side_b * B + side_c * C) / W_A if abs(W_A) > 1e-9 else A
    W_C = side_a + side_b - side_c
    J_C = (side_a * A + side_b * B - side_c * C) / W_C if abs(W_C) > 1e-9 else C

    return J_A, J_C

def get_user_input(prompt, default_value):
    user_str = input(f"{prompt} [直接按 Enter 使用預設值 {default_value}]: ").strip()
    if not user_str: return default_value
    try: return float(user_str)
    except ValueError: return default_value

def main():
    print("="*60)
    print("通用橢圓撞球模擬：相容橢圓與雙曲線焦散線")
    print("="*60)

    a0 = get_user_input("半長軸 a", 5.0)
    b0 = get_user_input("半短軸 b", 3.0)
    if a0 < b0: a0, b0 = b0, a0

    lam = get_user_input(f"焦散線參數 lambda (0 < lambda < {a0**2})", 4.0)
    iterations = int(get_user_input("反射次數", 1000))

    # 判斷焦散線類型
    is_hyperbola = lam > b0**2

    # 1. 產生初始切線路徑
    # 策略：在邊界上選一點，求解過該點且切於共焦曲線的斜率 m
    theta_start = np.random.uniform(0, 2*np.pi)
    x0, y0 = a0 * np.cos(theta_start), b0 * np.sin(theta_start)

    # 切線方程式滿足： (a^2-lam)m^2 + (b^2-lam) = (y0 - mx0)^2
    # 整理成 Am^2 + Bm + C = 0
    A_quad = (a0**2 - lam) - x0**2
    B_quad = 2 * x0 * y0
    C_quad = (b0**2 - lam) - y0**2

    roots = np.roots([A_quad, B_quad, C_quad])
    m = roots[0].real # 取其中一個切線斜率

    # 尋找下一個點 P_curr
    A_int = (b0**2 + a0**2 * m**2)
    B_int = 2 * a0**2 * m * (y0 - m * x0)
    C_int = a0**2 * (y0 - m * x0)**2 - a0**2 * b0**2
    x_roots = np.roots([A_int, B_int, C_int])
    x1 = x_roots[0] if abs(x_roots[0] - x0) > 1e-5 else x_roots[1]
    y1 = y0 + m * (x1 - x0)

    P_prev = np.array([x0, y0])
    P_curr = np.array([x1, y1])

    path_x, path_y = [P_prev[0], P_curr[0]], [P_prev[1], P_curr[1]]
    excenter_pts = []
    mags = []

    # 2. 執行模擬
    for _ in range(iterations):
        P_next = get_next_billiard_point(P_prev, P_curr, a0, b0)
        J_A, J_C = calculate_excenters(P_prev, P_curr, P_next)
        path_x.append(P_next[0]); path_y.append(P_next[1])
        excenter_pts.extend([J_A, J_C])
        mags.extend([np.linalg.norm(J_A/[a0, b0]), np.linalg.norm(J_C/[a0, b0])])
        P_prev, P_curr = P_curr, P_next

    # 3. 繪圖與視覺化
    fig, ax = plt.subplots(figsize=(11, 9))
    t = np.linspace(0, 2*np.pi, 500)
    ax.plot(a0 * np.cos(t), b0 * np.sin(t), 'k-', lw=2, label='Ellipse Boundary')

    if not is_hyperbola:
        # 繪製橢圓焦散線
        ac, bc = np.sqrt(a0**2 - lam), np.sqrt(b0**2 - lam)
        ax.plot(ac * np.cos(t), bc * np.sin(t), '--', color='orange', lw=2, label=f'Elliptic Caustic ($\lambda={lam}$)')
    else:
        # 繪製雙曲線焦散線
        ac, bc = np.sqrt(a0**2 - lam), np.sqrt(lam - b0**2)
        hyp_t = np.linspace(-2, 2, 500)
        ax.plot(ac * np.cosh(hyp_t), bc * np.sinh(hyp_t), '--', color='orange', lw=2, label=f'Hyperbolic Caustic ($\lambda={lam}$)')
        ax.plot(-ac * np.cosh(hyp_t), bc * np.sinh(hyp_t), '--', color='orange', lw=2)

    ax.plot(path_x, path_y, color='darkblue', linewidth=1.2, alpha=0.5, label='Trajectory')

    # 旁心點與最大值
    mags = np.array(mags)
    p_max = excenter_pts[np.argmax(mags)]
    ex_x, ex_y = zip(*excenter_pts)
    ax.scatter(ex_x, ex_y, s=0.5, color='red', alpha=0.2)
    ax.scatter(p_max[0], p_max[1], color='gold', marker='*', s=150, edgecolors='black', zorder=10, label=f'Max Scale: {np.max(mags):.3f}')

    # 理論圓形軌跡 (僅限橢圓焦散線參考，雙曲線時 Q 公式不同)
    if not is_hyperbola:
        D, c_sq = np.sqrt(lam), a0**2 - b0**2
        Q1 = (2 * a0**2 * b0 * D) / ((a0 * b0)**2 + c_sq * D**2)
        Q2 = (2 * a0 * b0**2 * D) / ((a0 * b0)**2 - c_sq * D**2)
        M1, M2 = np.sqrt(1 + Q1**2), np.sqrt(1 + Q2**2)

        # ================= 新增的列印區塊 =================
        print("\n" + "="*30)
        print(" 理論計算結果 (橢圓焦散線)")
        print("="*30)
        print(f"Q1 = {Q1:.6f}")
        print(f"Q2 = {Q2:.6f}")
        print(f"M1 = {M1:.6f}")
        print(f"M2 = {M2:.6f}")
        print("="*30 + "\n")
        # ==================================================

        ax.plot(M1 * a0 * np.cos(t), M1 * b0 * np.sin(t), 'b--', lw=1.5, label='Theory M1')
        ax.plot(M2 * a0 * np.cos(t), M2 * b0 * np.sin(t), 'm--', lw=1.5, label='Theory M2')

    ax.set_aspect('equal')
    ax.set_xlim(-a0*2, a0*2); ax.set_ylim(-b0*2, b0*2)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title(f"Confocal Billiard Simulation: $\lambda={lam}$ ({'Hyperbolic' if is_hyperbola else 'Elliptic'})")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()