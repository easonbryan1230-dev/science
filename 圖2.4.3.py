import numpy as np

def robust_excenter_solver():
    print("=== 橢圓旁心軌跡 (高精度掃描 + 嚴格 L_n 定義版) ===")
    a = float(input("請輸入原橢圓 a: "))
    b = float(input("請輸入原橢圓 b: "))
    lmbda = float(input("請輸入焦散參數 lambda: "))
    k = int(input("請輸入步長 k (頂點 B): "))
    d = int(input("請輸入步長 d (頂點 A, C): "))

    ac2, bc2 = a**2 - lmbda, b**2 - lmbda

    def get_tangent_vectors(P):
        x, y = P[0], P[1]
        A_c = ac2 - x**2
        B_c = 2*x*y
        C_c = bc2 - y**2

        if abs(A_c) < 1e-8:
            v1 = np.array([0.0, 1.0])
            v2 = np.array([1.0, -C_c / B_c])
        else:
            delta = max(0, B_c**2 - 4*A_c*C_c)
            m1 = (-B_c + np.sqrt(delta)) / (2*A_c)
            m2 = (-B_c - np.sqrt(delta)) / (2*A_c)
            v1 = np.array([1.0, m1])
            v2 = np.array([1.0, m2])

        return v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)

    def get_next_P(P, v):
        num = v[0]*P[0]/a**2 + v[1]*P[1]/b**2
        den = v[0]**2/a**2 + v[1]**2/b**2
        t = -2 * num / den
        return P + t * v

    def intersect_lines(p1, p2, p3, p4):
        a1, b1 = p2[1] - p1[1], p1[0] - p2[0]
        c1 = a1*p1[0] + b1*p1[1]
        a2, b2 = p4[1] - p3[1], p3[0] - p4[0]
        c2 = a2*p3[0] + b2*p3[1]
        det = a1*b2 - a2*b1
        if abs(det) < 1e-10: return np.array([1e9, 1e9])
        return np.array([(c1*b2 - c2*b1)/det, (a1*c2 - a2*c1)/det])

    def get_geometry(theta):
        P0 = np.array([a*np.cos(theta), b*np.sin(theta)])
        v1, v2 = get_tangent_vectors(P0)
        P1 = get_next_P(P0, v1)
        pts = [P0, P1]

        # 設定一個足夠大的初始指標 n，確保不會產生負數索引
        n = d

        # 確保光線陣列夠長 (要能涵蓋到 n+k)
        max_idx = n + k + 2
        for _ in range(max_idx):
            P_curr, P_prev = pts[-1], pts[-2]
            va, vb = get_tangent_vectors(P_curr)
            P_next_a = get_next_P(P_curr, va)
            P_next_b = get_next_P(P_curr, vb)

            if np.linalg.norm(P_next_a - P_prev) > np.linalg.norm(P_next_b - P_prev):
                pts.append(P_next_a)
            else:
                pts.append(P_next_b)

        # --- 完全依照你的幾何邏輯 ---
        # B = L_n 交 L_{n+k}
        B = intersect_lines(pts[n], pts[n+1], pts[n+k], pts[n+k+1])
        # A = L_n 交 L_{n+d}
        A = intersect_lines(pts[n], pts[n+1], pts[n+d], pts[n+d+1])
        # C = L_{n+k} 交 L_{n+k-d}
        C = intersect_lines(pts[n+k], pts[n+k+1], pts[n+k-d], pts[n+k-d+1])

        # 計算 A 側旁心 Ja (對立於邊 BC)
        la, lb, lc = np.linalg.norm(B-C), np.linalg.norm(A-C), np.linalg.norm(A-B)
        Ja = (-la*A + lb*B + lc*C) / (-la + lb + lc)
        return B, Ja

    print("正在掃描 360 度軌跡以尋找精確頂點...")
    t_vals = np.linspace(0, 2*np.pi, 2000)
    B_pts = []
    Ja_pts = []

    for t in t_vals:
        try:
            B, Ja = get_geometry(t)
            if abs(B[0]) < 1000:
                B_pts.append(B)
                Ja_pts.append(Ja)
        except:
            continue

    B_pts = np.array(B_pts)
    Ja_pts = np.array(Ja_pts)

    idx_long = np.argmax(np.abs(B_pts[:, 0]))
    idx_short = np.argmax(np.abs(B_pts[:, 1]))

    ak = np.abs(B_pts[idx_long, 0])
    bk = np.abs(B_pts[idx_short, 1])

    Ye = np.abs(Ja_pts[idx_long, 1])
    Xe = np.abs(Ja_pts[idx_short, 0])

    delta = ak**2 * bk**2 - Xe**2 * Ye**2
    C1 = (bk**2 - Ye**2) / delta
    C2 = (ak**2 - Xe**2) / delta

    print(f"\n--- [完美對應版] 模型參數提取 ---")
    print(f"第 {k} 階橢圓頂點: a_k = {ak:.6f}, b_k = {bk:.6f}")
    print(f"B在長軸時，A側旁心座標: 假設 ({ak:.6f}, {Ye:.6f})")
    print(f"B在短軸時，A側旁心座標: 假設 ({Xe:.6f}, {bk:.6f})")

    print(f"\n--- 聯立方程解果 ---")
    print(f"軌跡係數 C1 = {C1:.10f}")
    print(f"軌跡係數 C2 = {C2:.10f}")
    print(f"長半軸平方 (1/C1) = {1/C1:.6f}")
    print(f"短半軸平方 (1/C2) = {1/C2:.6f}")
    print(f"最終方程: {C1:.8f}x^2 + {C2:.8f}y^2 = 1")

robust_excenter_solver()