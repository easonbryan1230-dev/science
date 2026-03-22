import numpy as np
import matplotlib.pyplot as plt

def robust_excenter_solver():
    print("=== 橢圓旁心軌跡 (高精度 1500 點採樣版) ===")
    a = float(input("請輸入原橢圓 a: "))
    b = float(input("請輸入原橢圓 b: "))
    lmbda = float(input("請輸入焦散參數 lambda: "))
    k = int(input("請輸入步長 k (頂點 B): "))
    d = int(input("請輸入步長 d (頂點 A, C): "))

    ac2, bc2 = a**2 - lmbda, b**2 - lmbda
    
    if ac2 <= 0 or bc2 <= 0:
        print("錯誤：lambda 太大，導致焦散橢圓不存在！")
        return

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
        
        n = max(abs(d), abs(k), abs(k-d)) + 2 
        
        idx_B1, idx_B2 = n, n + k
        idx_A1, idx_A2 = n, n + d
        idx_C1, idx_C2 = n + k, n + k - d
        
        max_idx = max(idx_B1, idx_B2, idx_A1, idx_A2, idx_C1, idx_C2) + 2
        
        for _ in range(max_idx):
            P_curr, P_prev = pts[-1], pts[-2]
            va, vb = get_tangent_vectors(P_curr)
            P_next_a = get_next_P(P_curr, va)
            P_next_b = get_next_P(P_curr, vb)
            
            if np.linalg.norm(P_next_a - P_prev) > np.linalg.norm(P_next_b - P_prev):
                pts.append(P_next_a)
            else:
                pts.append(P_next_b)
        
        B = intersect_lines(pts[idx_B1], pts[idx_B1+1], pts[idx_B2], pts[idx_B2+1])
        A = intersect_lines(pts[idx_A1], pts[idx_A1+1], pts[idx_A2], pts[idx_A2+1])
        C = intersect_lines(pts[idx_C1], pts[idx_C1+1], pts[idx_C2], pts[idx_C2+1])
        
        la, lb, lc = np.linalg.norm(B-C), np.linalg.norm(A-C), np.linalg.norm(A-B)
        Ja = (-la*A + lb*B + lc*C) / (-la + lb + lc)
        return B, Ja

    print("\n正在掃描 360 度軌跡以尋找精確頂點與繪圖點 (1500點)...")
    # === 這裡改成 1500 點 ===
    t_vals = np.linspace(0, 2*np.pi, 1500)
    B_pts = []
    Ja_pts = []
    
    for t in t_vals:
        try:
            B, Ja = get_geometry(t)
            if abs(B[0]) < 1000 and not np.isnan(Ja).any():
                B_pts.append(B)
                Ja_pts.append(Ja)
        except:
            continue
            
    if len(B_pts) == 0:
        print("無法計算出有效軌跡，請檢查參數組合。")
        return

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

    print(f"\n--- 模型參數提取 ---")
    print(f"第 {k} 階橢圓頂點: a_k = {ak:.6f}, b_k = {bk:.6f}")
    print(f"B在長軸時，A側旁心座標: ({ak:.6f}, {Ye:.6f})")
    print(f"B在短軸時，A側旁心座標: ({Xe:.6f}, {bk:.6f})")
    
    print(f"\n--- 聯立方程解果 ---")
    print(f"軌跡係數 C1 = {C1:.10f}")
    print(f"軌跡係數 C2 = {C2:.10f}")
    if C1 > 0 and C2 > 0:
        print(f"旁心軌跡長半軸 = {np.sqrt(1/C1):.6f}")
        print(f"旁心軌跡短半軸 = {np.sqrt(1/C2):.6f}")
    print(f"最終方程: {C1:.8f}x^2 + {C2:.8f}y^2 = 1")

    # ================= 繪圖部分 =================
    print("\n正在生成圖形，請稍候...")
    plt.figure(figsize=(8, 8))
    theta_plot = np.linspace(0, 2*np.pi, 300)
    
    plt.plot(a * np.cos(theta_plot), b * np.sin(theta_plot), 'k--', label='Original Ellipse')
    
    a_c = np.sqrt(ac2)
    b_c = np.sqrt(bc2)
    plt.plot(a_c * np.cos(theta_plot), b_c * np.sin(theta_plot), color='gray', linestyle=':', label='Caustic Ellipse')
    
    plt.plot(ak * np.cos(theta_plot), bk * np.sin(theta_plot), 'b-', linewidth=1.5, label=f'B Locus ({k}-th Order)')
    # 橘點也變多，稍微把點縮小一點 (s=3) 會比較好看
    plt.scatter(B_pts[:, 0], B_pts[:, 1], s=3, c='cyan', zorder=3) 
    
    if C1 > 0 and C2 > 0:
        a_ja = np.sqrt(1 / C1)
        b_ja = np.sqrt(1 / C2)
        plt.plot(a_ja * np.cos(theta_plot), b_ja * np.sin(theta_plot), 'r-', linewidth=2, label='Excenter Locus (Ja)')
        plt.scatter(Ja_pts[:, 0], Ja_pts[:, 1], s=3, c='orange', zorder=3)
    else:
        print("注意：計算出的 C1 或 C2 為負數，旁心軌跡可能不是封閉橢圓或出現計算誤差。")

    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.title(f'Poncelet Polygon & Excenter Locus (k={k}, d={d})')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()

robust_excenter_solver()