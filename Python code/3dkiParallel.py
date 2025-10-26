# loop k

import os
import math
import time
import numpy as np
from numba import njit
from multiprocessing import Pool, cpu_count


class JavaRandom:
    def __init__(self, seed: int):
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    def next(self, bits: int) -> int:
        self.seed = (self.seed * 25214903917 + 11) & ((1 << 48) - 1)
        return self.seed >> (48 - bits)
    def next_float(self) -> float:
        return self.next(24) / float(1 << 24)

# =========================
# parameter
# =========================
NUM         = 100                     #particle number
Size        = 400.0                   #L
nyu         = 0.20                    #noise parameter η
alpha       = 1.5                     #distance-decay attenuation
Eps         = 1e-5                    #epsilon
RUN_FRAMES  = 10000                   #time scale
RS_START    = 1                       #Randomseed start
RS_END      = 100                     #Randomseed end
K_LIST      = [1,2,3,4,5,6,7,8,9,10]  #List of average number of neighbor
NOISE_SCALE = 0.5                     #noise scale
SAVE_DIR    = "" #folder place here

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_filename(size_val: float, Qval: float, nyu_val: float, seed_val: int, alpha_val: float, effect_num: int) -> str:
    q_str   = f"{Qval:.2f}"
    nyu_str = f"{nyu_val:.2f}"
    base = f"{int(size_val)}({q_str},{nyu_str},{seed_val},{alpha_val}){effect_num}.txt"
    return os.path.join(SAVE_DIR, base)

def precompute_randoms(rs: int, frames: int, n: int):
    rng = JavaRandom(rs)
    def r(): return rng.next_float()

    px = np.empty(n, dtype=np.float32)
    py = np.empty(n, dtype=np.float32)
    pz = np.empty(n, dtype=np.float32)
    th = np.empty(n, dtype=np.float32)
    gm = np.empty(n, dtype=np.float32)

    for i in range(n):
        px[i] = r() * Size - Size/2
        py[i] = r() * Size - Size/2
        pz[i] = r() * Size - Size/2
        th[i] = r() * 2.0 * math.pi
        gm[i] = r() * 2.0 * math.pi

    Xi = np.empty((frames, n), dtype=np.float32)
    p1 = np.empty((frames, n), dtype=np.float32)
    p2 = np.empty((frames, n), dtype=np.float32)
    q1 = np.empty((frames, n), dtype=np.float32)
    q2 = np.empty((frames, n), dtype=np.float32)

    for t in range(frames):
        for i in range(n):
            Xi[t, i] = (r() * 2.0 - 1.0) * math.pi
            p1[t, i] = r()
            p2[t, i] = r()
            q1[t, i] = r()
            q2[t, i] = r()

    return Xi, p1, p2, q1, q2, th, gm, px, py, pz

@njit(cache=True)
def f11(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return nx*nx*one_c + c
@njit(cache=True)
def f12(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return nx*ny*one_c - nz*s
@njit(cache=True)
def f13(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return nx*nz*one_c + ny*s
@njit(cache=True)
def f21(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return ny*nx*one_c + nz*s
@njit(cache=True)
def f22(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return ny*ny*one_c + c
@njit(cache=True)
def f23(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return ny*nz*one_c - nx*s
@njit(cache=True)
def f31(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return nz*nx*one_c - ny*s
@njit(cache=True)
def f32(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return nz*ny*one_c + nx*s
@njit(cache=True)
def f33(nx, ny, nz, theta):
    c, s, one_c = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return nz*nz*one_c + c

@njit(cache=True)
def select_k_smallest_indices_stable(d_arr, K):
    n = d_arr.shape[0]
    out = np.empty(K, dtype=np.int32)
    used = np.zeros(n, dtype=np.uint8)
    for k in range(K):
        best_j = -1
        best_d = 1e30
        for j in range(n):
            if used[j] == 1:
                continue
            dj = d_arr[j]
            if dj < best_d or (dj == best_d and (best_j == -1 or j < best_j)):
                best_d = dj
                best_j = j
        out[k] = best_j
        used[best_j] = 1
    return out

@njit(cache=True)
def step_once(px, py, pz, vx, vy, vz,
              Size, R, effectK,
              Xi_row, p1_row, p2_row, q1_row, q2_row,
              nyu, Q, alpha, Eps) -> np.float32:

    n = px.shape[0]
    new_vx = np.empty(n, dtype=np.float32)
    new_vy = np.empty(n, dtype=np.float32)
    new_vz = np.empty(n, dtype=np.float32)

    Size32 = np.float32(Size)
    half   = np.float32(Size/2.0)
    Rf     = np.float32(R)
    EPS    = np.float32(1e-12)

    for i in range(n):
        # ---- metric----
        sumMx, sumMy, sumMz = vx[i], vy[i], vz[i]
        countM = np.float32(1.0)
        d_arr = np.empty(n, dtype=np.float32)

        for j in range(n):
            if j == i:
                d_arr[j] = np.float32(1e30)
                continue
            dx = px[i] - px[j]
            if dx >  half: dx -= Size32
            elif dx < -half: dx += Size32
            dy = py[i] - py[j]
            if dy >  half: dy -= Size32
            elif dy < -half: dy += Size32
            dz = pz[i] - pz[j]
            if dz >  half: dz -= Size32
            elif dz < -half: dz += Size32

            d1 = np.float32(math.sqrt(dx*dx + dy*dy + dz*dz))
            d_arr[j] = d1

            if d1 > 0.0 and d1 < Rf:
                wden = np.float32(1.0) if alpha == 0.0 else (d1 ** alpha) + np.float32(Eps)
                sumMx += vx[j] / wden
                sumMy += vy[j] / wden
                sumMz += vz[j] / wden
                countM += np.float32(1.0)

        invM = np.float32(1.0) / countM
        Nx = sumMx * invM; Ny = sumMy * invM; Nz = sumMz * invM
        Nm = np.float32(math.sqrt(Nx*Nx + Ny*Ny + Nz*Nz))
        if Nm < EPS: Nm = EPS
        Nx /= Nm; Ny /= Nm; Nz /= Nm

        Xi = np.float32(NOISE_SCALE * nyu) * Xi_row[i] 
 
        rx = 2.0 * p1_row[i] - 1.0
        ry = 2.0 * p2_row[i] - 1.0
        rz = 2.0 * q1_row[i] - 1.0
        dot = rx*Nx + ry*Ny + rz*Nz
        kx = rx - dot*Nx
        ky = ry - dot*Ny
        kz = rz - dot*Nz
        km = math.sqrt(kx*kx + ky*ky + kz*kz) + 1e-12
        kx /= km; ky /= km; kz /= km

        ct = math.cos(Xi); st = math.sin(Xi); oc = 1.0 - ct
        cx = ky*Nz - kz*Ny
        cy = kz*Nx - kx*Nz
        cz = kx*Ny - ky*Nx
        dot2 = kx*Nx + ky*Ny + kz*Nz
        vMx = Nx*ct + cx*st + kx*dot2*oc
        vMy = Ny*ct + cy*st + ky*dot2*oc
        vMz = Nz*ct + cz*st + kz*dot2*oc

        # ---- topological----
        K = effectK if effectK < n else (n - 1)
        idx = select_k_smallest_indices_stable(d_arr, K)

        sumTx, sumTy, sumTz = vx[i], vy[i], vz[i]
        countT = np.float32(1.0)
        for k in range(K):
            j = idx[k]; dj = d_arr[j]
            wden = np.float32(1.0) if alpha == 0.0 else (dj ** alpha) + np.float32(Eps)
            sumTx += vx[j] / wden
            sumTy += vy[j] / wden
            sumTz += vz[j] / wden
            countT += np.float32(1.0)

        invT = np.float32(1.0) / countT
        Tx = sumTx * invT; Ty = sumTy * invT; Tz = sumTz * invT
        Tm = np.float32(math.sqrt(Tx*Tx + Ty*Ty + Tz*Tz))
        if Tm < EPS: Tm = EPS
        Tx /= Tm; Ty /= Tm; Tz /= Tm


        Xi = np.float32(NOISE_SCALE * nyu) * Xi_row[i]
        rx = 2.0 * q1_row[i] - 1.0
        ry = 2.0 * q2_row[i] - 1.0
        rz = 2.0 * p1_row[i] - 1.0
        dot = rx*Tx + ry*Ty + rz*Tz
        kx = rx - dot*Tx
        ky = ry - dot*Ty
        kz = rz - dot*Tz
        km = math.sqrt(kx*kx + ky*ky + kz*kz) + 1e-12
        kx/=km; ky/=km; kz/=km
        ct = math.cos(Xi); st = math.sin(Xi); oc = 1.0 - ct
        cx = ky*Tz - kz*Ty
        cy = kz*Tx - kx*Tz
        cz = kx*Ty - ky*Tx
        dot2 = kx*Tx + ky*Ty + kz*Tz
        vTx = Tx*ct + cx*st + kx*dot2*oc
        vTy = Ty*ct + cy*st + ky*dot2*oc
        vTz = Tz*ct + cz*st + kz*dot2*oc


        nVx = (1.0-Q)*vMx + Q*vTx
        nVy = (1.0-Q)*vMy + Q*vTy
        nVz = (1.0-Q)*vMz + Q*vTz
        nm  = np.float32(math.sqrt(nVx*nVx + nVy*nVy + nVz*nVz))
        if nm < EPS: nm = EPS
        new_vx[i] = nVx / nm; new_vy[i] = nVy / nm; new_vz[i] = nVz / nm


    for i in range(n):
        vx[i]=new_vx[i]; vy[i]=new_vy[i]; vz[i]=new_vz[i]
        px[i]+=vx[i]; py[i]+=vy[i]; pz[i]+=vz[i]
        half2 = Size32/2
        if px[i] < -half2: px[i]+=Size32
        elif px[i] > half2: px[i]-=Size32
        if py[i] < -half2: py[i]+=Size32
        elif py[i] > half2: py[i]-=Size32
        if pz[i] < -half2: pz[i]+=Size32
        elif pz[i] > half2: pz[i]-=Size32


    sx=sy=sz=0.0
    for i in range(n):
        sx+=vx[i]; sy+=vy[i]; sz+=vz[i]
    m = np.float32(math.sqrt(sx*sx+sy*sy+sz*sz))
    return np.float32(m/np.float32(n))


def run_one(RS: int, effectK: int, Q: float) -> None:
    ensure_dir(SAVE_DIR)
    t0 = time.perf_counter()
    R = Size * ((3.0 * effectK) / (4.0 * math.pi * NUM)) ** (1.0/3.0)
    Xi, p1, p2, q1, q2, th, gm, px, py, pz = precompute_randoms(RS, RUN_FRAMES, NUM)
    vx = (np.sin(th)*np.cos(gm)).astype(np.float32)
    vy = (np.sin(th)*np.sin(gm)).astype(np.float32)
    vz = (np.cos(th)).astype(np.float32)
    saveData=[]
    sx,sy,sz=vx.sum(),vy.sum(),vz.sum()
    va0=np.float32(math.sqrt(sx*sx+sy*sy+sz*sz)/NUM)
    saveData.append(f"{va0:.8f}")
    for t in range(RUN_FRAMES):
        va=step_once(px,py,pz,vx,vy,vz,Size,R,effectK,Xi[t],p1[t],p2[t],q1[t],q2[t],nyu,Q,alpha,Eps)
        if (t+1)%100==0: saveData.append(f"{va:.8f}")
        if not (va>0.0): break
    elapsed=time.perf_counter()-t0
    fn=make_filename(Size,Q,nyu,RS,alpha,effectK)
    with open(fn,"w") as f: f.write("\n".join(saveData))
    print(f"[Q={Q:.2f}] Saved: {fn} ({len(saveData)} rows, time={elapsed:.2f}s)")


def run_all_for_Q(Q: float):
    for RS in range(RS_START, RS_END+1):
        for k in K_LIST:
            run_one(RS, int(k), Q)

def main():
    start=time.time()
    Q_LIST=[round(0.1*i,2) for i in range(0,11)]
    NUM_WORKERS=min(len(Q_LIST),max(1,cpu_count()-2))
    print(f"Start parallel by Q: workers={NUM_WORKERS}, Q_LIST={Q_LIST}")
    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(run_all_for_Q,Q_LIST)
    print(f"All runs finished. Total elapsed: {time.time()-start:.2f}s")

if __name__=="__main__":
    main()
