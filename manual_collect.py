# -*- coding: utf-8 -*-
"""手动（遥操作）采集界面：键鼠开机器人抓物块，边操作边录示范。

对应教程里的 `0.teleop.ipynb` / `collect_data.py`（人用 leader 臂遥操作），
只不过这里不接真机手柄，而是直接键盘驱动仿真里的齿面中点：

    W/S          齿面中点沿 +x / -x 平移（远离 / 靠近机器人底座）
    A/D          沿 +y / -y 平移（左/右）
    Q/E          沿 +z / -z 平移（升/降）
    Space        夹爪开/合切换（合爪用按物块宽度标定的夹持角，会真的夹住）
    T            重来：同一组布局，臂回预备位姿、本段录制清空
    R            换一组随机布局（物块位置 + 放置点）
    [ / ]        调慢 / 调快移动速度
    Enter        结束本段：判定成功则存入数据集
    Backspace    放弃本段并换一组布局
    Esc          退出（也可直接关窗口）

对位余量：遥操作时钳口开得比专家脚本更宽（默认净距 = 物块边长 + 36mm，每边 18mm 余量），
因为人是手动对位，而专家脚本是 IK 精确对位（它只要 14mm）。
另外"开始移动之后"才开始录，开局发呆的帧不会进数据集。

运动学：每拍把齿面中点目标 p_target 做几步齿面中点 IK（solve_ik_grasp），
再把得到的关节目标下发给 8 个位置执行器、跑一个控制周期（默认 50ms）。
手感与专家脚本一致（同一套 IK / 同一个夹爪标定），所以录出来的数据可以和专家数据
直接混在一起训练。

录制约定与 src/expert_grasp.py 完全一致，所以 train.py / eval_grasp.py 不用改：
    obs    = env.get_obs()                                     17 维
    action = [未来 lookahead 拍后的实测关节角(6), 该拍下发的夹爪命令(1)]  7 维

用法：
    python manual_collect.py                     # 打开 MuJoCo 窗口，键盘遥操作
    python manual_collect.py --out_dir data/manual --episodes 20
    python manual_collect.py --selftest          # 不开窗口：用脚本化的"假人"跑完整流程，验证录制/存盘
"""
import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk

import mujoco
from src.env_grasp import UranusGraspEnv
from src.expert_grasp import DEFAULT_LOOKAHEAD, MIN_LIFT, build_frames


# 移动键 -> (世界坐标轴, 符号)。注意：这里用的是**我们自己 OpenCV 窗口**的按键，
# 不再经过 MuJoCo 查看器 —— 查看器自带一堆可视化快捷键（W=网格化、S=亮度、A=自动连接、
# D=骨架、Q=相机、E=等价约束、R=反射、T=透明…），WASD 会边操作边把渲染开关按个遍，
# 表现就是"背景忽明忽暗、模型闪烁"。换成自己的窗口后键位完全由我们控制。
KEYMAP = {
    "w": (0, +1.0), "s": (0, -1.0),     # ±x
    "a": (1, +1.0), "d": (1, -1.0),     # ±y
    "q": (2, +1.0), "e": (2, -1.0),     # ±z
}


# 齿面中点允许活动的范围（别让遥操作把手臂开进桌面/开出工作空间）
LIMITS = {
    "x": (1.42, 1.80),
    "y": (-0.26, 0.22),
    "z": (0.365, 0.78),
}

KEYHELP = """\
按键：
  W/S   齿面中点沿 +x / -x（远离 / 靠近机器人底座）
  A/D   沿 +y / -y（左 / 右）
  Q/E   沿 +z / -z（升 / 降）
  Space 夹爪开 / 合
  T     重来：同一组布局，臂回预备位姿、本段录制清空
  R     换一组新布局（丢弃当前录制）
  [ ]   调慢 / 调快速度
  Enter 手动结算本段        Backspace 放弃本段        Esc 退出

怎么算成功：方块落到蓝色圆盘上、夹爪张开、稳定 0.4s -> 自动存盘并换新布局。
对位提示：夹爪沿 y 开合，y 方向要对准（4cm 方块 + 30mm 富余量 -> 每边约 1.5cm 余量）；
         x 方向齿条 7cm 长，容忍 ±1.6cm；z 让齿咬住方块中间高度即可。"""


class TkView:
    """用 tkinter 窗口显示 MuJoCo 离屏渲染的画面。

    为什么不用 mujoco.viewer：查看器自带大量可视化快捷键（W=网格化、S=亮度、A=自动连接、
    D=骨架、Q=相机、E=等价约束、R=反射、T=透明…），和遥操作的移动键全线撞车，边操作边把
    渲染开关按乱 —— 表现就是"背景忽明忽暗、模型闪烁"。
    为什么不用 cv2.imshow：这个 conda 环境里的 opencv 是 headless 构建，imshow 未实现。
    tkinter 是标准库、走 GDI、不碰 OpenGL，远程桌面下也不会黑帧，而且能用系统字体写中文 HUD。
    """

    def __init__(self, model, width: int, height: int, cam, title="Uranus 遥操作采集"):
        offw = int(model.vis.global_.offwidth)
        offh = int(model.vis.global_.offheight)
        width, height = min(width, offw), min(height, offh)
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = cam
        self.events = collections.deque()
        self._drag = None
        self._photo = None

        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.events.append(("close", None)))
        self.label = tk.Label(self.root, borderwidth=0)
        self.label.pack()
        # 按下/松开都收：这样"按住某个方向键"就是连续运动（不再依赖系统的按键重复）
        self.root.bind("<KeyPress>", lambda e: self.events.append(("press", e.keysym)))
        self.root.bind("<KeyRelease>", lambda e: self.events.append(("release", e.keysym)))
        self.root.bind("<FocusOut>", lambda e: self.events.append(("focusout", None)))
        self.label.bind("<ButtonPress-1>", self._mouse_down)
        self.label.bind("<B1-Motion>", self._mouse_move)
        self.label.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        self.label.bind("<ButtonPress-3>", self._mouse_down)
        self.label.bind("<B3-Motion>", self._mouse_move)
        self.label.bind("<ButtonRelease-3>", lambda e: setattr(self, "_drag", None))
        self.root.bind("<MouseWheel>", self._mouse_wheel)
        self.label.focus_set()
        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
        except OSError:
            self.font = ImageFont.load_default()
        self.root.update()

    # ---- 鼠标：左键拖动=转视角，右键拖动=平移，滚轮=远近 ----
    def _mouse_down(self, event):
        self._drag = (event.x, event.y, "orbit" if event.num == 1 else "pan")
        self.label.focus_set()

    def _mouse_move(self, event):
        if self._drag is None:
            return
        px, py, mode = self._drag
        dx, dy = event.x - px, event.y - py
        if mode == "orbit":
            self.cam.azimuth -= dx * 0.4
            self.cam.elevation = float(np.clip(self.cam.elevation - dy * 0.4, -89, 89))
        else:
            a = np.deg2rad(self.cam.azimuth)
            right = np.array([np.cos(a), np.sin(a), 0.0])
            self.cam.lookat[:] = self.cam.lookat - dx * 0.0015 * right + dy * 0.0015 * np.array([0, 0, 1.0])
        self._drag = (event.x, event.y, mode)

    def _mouse_wheel(self, event):
        self.cam.distance = float(np.clip(self.cam.distance * (0.9 if event.delta > 0 else 1.1), 0.4, 6.0))

    def poll_event(self):
        return self.events.popleft() if self.events else None

    def draw(self, data, hud_lines):
        self.renderer.update_scene(data, camera=self.cam)
        img = Image.fromarray(self.renderer.render())
        d = ImageDraw.Draw(img, "RGBA")
        for i, line in enumerate(hud_lines):
            y = 8 + 24 * i
            w = d.textlength(line, font=self.font)
            d.rectangle([6, y, 16 + w, y + 22], fill=(0, 0, 0, 150))
            d.text((10, y + 2), line, font=self.font, fill=(255, 255, 255, 255))
        self._photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)
        self.root.update()

    @property
    def alive(self) -> bool:
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


class TeleopCollector:
    def __init__(self, env: UranusGraspEnv, args):
        self.env = env
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.keymap = dict(KEYMAP)

        # 把遥操作的钳口开得比专家脚本更宽，给人的手动对位留余量
        env.open_angle = float(env.open_angle_for(margin=args.open_margin))

        self.p_target = None
        self.gripper_open = True
        self.held = set()             # 当前按住的移动键（按住=连续走）
        self.last_press = {}          # 按键 -> 时间戳（松开事件丢失时的兜底）
        self.toggle_debounce = {}     # 按键 -> 时间戳（避免按键重复把开关抖成乱码）
        self.speed = args.speed

        self.obs_list, self.act_list, self.ep_ends = [], [], []
        self.stream = []
        self.step_count = 0
        self.ep_obj0 = None
        self.ep_place = None
        self.ep_peaks = []
        self.ep_saved = False
        self.success_since = None
        self.saved_episodes = 0
        self.quit = False
        self.message = ""

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------
    def _press_ok(self, keycode, dt=0.35):
        """同一个键在 dt 秒内的重复事件忽略（按住不放时系统会连发按键重复）。"""
        now = time.time()
        if now - self.toggle_debounce.get(keycode, -9.0) < dt:
            return False
        self.toggle_debounce[keycode] = now
        return True

    def handle_event(self, kind: str, value):
        """press / release / focusout / close 四种事件。"""
        if kind == "close":
            self.message = "窗口被关闭，退出"
            self.quit = True
            return
        if kind == "focusout":
            self.held.clear()        # 失焦时把"按住"清掉，免得键卡住一直走
            return
        low = (value or "").lower()
        if kind == "release":
            self.held.discard(low)
            return
        self.handle_key(low)

    def handle_key(self, key: str):
        """处理一次按下（tkinter 的 keysym，比如 'w' / 'space' / 'Return'）。"""
        if not key:
            return
        low = key.lower()
        if low in self.keymap:
            self.last_press[low] = time.time()
            self.held.add(low)
        elif low == "space" and self._press_ok("space"):
            self.gripper_open = not self.gripper_open
            self.message = f"夹爪{'张开' if self.gripper_open else '合上'}"
        elif low == "r" and self._press_ok("r"):
            self.new_episode()
            self.message = "换了一组布局（当前录制已丢弃）"
        elif low == "t" and self._press_ok("t"):
            obj_pose, place = self.last_layout
            self.new_episode(obj_pose=obj_pose, place=place)
            self.message = "重来：同一组布局，臂已回预备位姿"
        elif low in ("return", "kp_enter") and self._press_ok("enter"):
            self.finish_episode(keep=True)
        elif low == "backspace" and self._press_ok("back"):
            self.new_episode()
            self.message = "已放弃本段"
        elif low == "bracketleft" and self._press_ok("["):
            self.speed = max(0.02, self.speed / 1.5)
            self.message = f"速度 {self.speed:.3f} m/s"
        elif low == "bracketright" and self._press_ok("]"):
            self.speed = min(0.60, self.speed * 1.5)
            self.message = f"速度 {self.speed:.3f} m/s"
        elif low == "escape":
            self.quit = True

    def move_target(self, dt: float):
        """把按住的键积分成齿面中点的目标位置。"""
        if self.p_target is None:
            return
        now = time.time()
        step = np.zeros(3)
        for key, (axis, sign) in self.keymap.items():
            # 主要看"是否按住"；另外保留 0.25s 闪现窗口，防止松开事件丢失导致不动
            if key in self.held or now - self.last_press.get(key, -9.0) < self.args.hold:
                step[axis] += sign
        if np.any(step):
            self.p_target = self.p_target + step * self.speed * dt
            self.p_target[0] = np.clip(self.p_target[0], *LIMITS["x"])
            self.p_target[1] = np.clip(self.p_target[1], *LIMITS["y"])
            self.p_target[2] = np.clip(self.p_target[2], *LIMITS["z"])

    # ------------------------------------------------------------------
    # 一个控制周期
    # ------------------------------------------------------------------
    def control_tick(self, target=None):
        env = self.env
        dt = self.args.ctrl_steps * env.model.opt.timestep     # 本拍推进的仿真时间（秒）
        p = self.p_target if target is None else np.asarray(target, dtype=float)
        if getattr(self, "_prev_p", None) is not None:         # 统计本段工具走了多远（自查用）
            self.ep_travel += float(np.linalg.norm(p - self._prev_p))
        self._prev_p = p.copy()
        q, _ = env.solve_ik_grasp(p, q0=env.get_joints(), max_iter=self.args.ik_iters)
        # 夹爪：闭合可以快（对称夹住），张开必须限速——瞬开会把物块弹飞（专家脚本里实测 17cm）。
        # 限速后"慢慢张开"这段斜坡会像专家数据一样被录进 action，策略学得到。
        want = float(env._grip_ctrl(self.gripper_open)[0])
        if want > self.jaw_cmd:
            self.jaw_cmd = min(want, self.jaw_cmd + self.args.jaw_rate * dt)
        else:
            self.jaw_cmd = want
        jaw = self.jaw_cmd
        env.data.ctrl[:] = np.concatenate([q, [jaw, jaw]])
        # 等你真的把夹爪挪动了(>5mm)才开始录，省掉开局发呆/对位之前的帧
        if not self.recording and np.linalg.norm(env.tooth_midpoint() - self.home_target) > 0.005:
            self.recording = True
            self.step_count = 0
            self.stream = []
        for _ in range(self.args.ctrl_steps):
            mujoco.mj_step(env.model, env.data)
            if not self.recording:
                continue
            self.step_count += 1
            if self.step_count % self.args.rec_every == 0:
                self.stream.append((env.get_obs().astype(np.float32).copy(),
                                    env.get_joints().astype(np.float32), jaw))

    # ------------------------------------------------------------------
    # episode 生命周期
    # ------------------------------------------------------------------
    def new_episode(self, obj_pose=None, place=None):
        if obj_pose is None or place is None:
            obj_pose, place = self.env.sample_layout(self.rng)
        self.last_layout = (obj_pose, place)
        self.env.reset(object_pose=obj_pose)
        self.env.place_on_table(place)
        self.p_target = self.env.tooth_midpoint().copy()
        self.home_target = self.p_target.copy()
        self.gripper_open = True
        self.jaw_cmd = float(self.env.open_angle)      # 实际下发的夹爪角（张开时限速）
        self.stream = []
        self.step_count = 0
        self.recording = False        # 等你真的开始动才录，省掉前面发呆的帧
        self.ep_obj0 = self.env.get_object_pos().copy()
        self.ep_place = place
        self.ep_peaks = [self.ep_obj0[2]]
        self.ep_saved = False
        self.success_since = None
        self.ep_frames = 0
        self.ep_travel = 0.0          # 本段工具点走过的路程（用来自查"到底动没动"）
        self._prev_p = None

    def is_placed(self) -> bool:
        """任务真的完成了：物块落在放置点附近、夹爪张开，而且物块已经**落在桌面上**
        （不能只看 env.success()——举着物块从放置点上方经过时它也成立，会误判成完成）。"""
        env = self.env
        rest_z = 0.35 + env.object_half
        return bool(env.success() and abs(env.get_object_pos()[2] - rest_z) < 0.012)

    def maybe_autosave(self, now: float) -> bool:
        """任务完成后自动结算这一段（连续成立 auto_hold 秒才算，避免手抖误触发）。"""
        if not self.args.auto_success:
            return False
        if self.ep_saved or len(self.stream) < self.args.min_frames:
            return False
        if self.is_placed():
            if self.success_since is None:
                self.success_since = now
            elif now - self.success_since >= self.args.auto_hold:
                print(f"  [自动保存] 检测到放置完成（保持 {self.args.auto_hold:.1f}s）")
                self.finish_episode(keep=True, auto=True)
                return True
        else:
            self.success_since = None
        return False

    def finish_episode(self, keep: bool, auto: bool = False):
        res = self.env
        lift = float(max(self.ep_peaks) - self.ep_obj0[2]) if self.ep_peaks else 0.0
        err = float(np.linalg.norm(res.get_object_pos()[:2] - self.ep_place[:2]))
        ok = bool(res.success() and lift > MIN_LIFT)
        why = "" if ok else ("物块没放到蓝盘上" if not res.success() else
                             f"只推没拿起来（抬升 {lift * 1000:.0f}mm < {MIN_LIFT * 1000:.0f}mm）")
        print(f"  本段：抬升 {lift * 1000:+.0f}mm  放置误差 {err * 1000:.0f}mm  "
              f"工具走了 {self.ep_travel * 100:.0f}cm  录到 {len(self.stream)} 帧  "
              f"success={res.success()}  ->  {'成功' if ok else '未完成：' + why}"
              f"{'（自动）' if auto else ''}", flush=True)
        if keep and self.stream:
            if not ok and not self.args.keep_failed:
                print("  （没完成，不写进数据集；想强制保存加 --keep_failed）")
                self.new_episode()
                return
            frames = build_frames(self.stream, self.args.lookahead)
            self.last_frames = frames
            self.last_episode = (self.last_layout[0], self.last_layout[1], frames)
            for obs, act in frames:
                self.obs_list.append(obs)
                self.act_list.append(act)
            self.ep_ends.append(len(self.obs_list))
            self.saved_episodes += 1
            self.ep_saved = True
            self.save()
            print(f"  已保存：episode {self.saved_episodes}/{self.args.episodes}，{len(frames)} 帧，"
                  f"累计 {len(self.obs_list)} 帧 -> {self.out_path()}", flush=True)
            if self.saved_episodes >= self.args.episodes:
                print(f"\n已经录满 {self.args.episodes} 段，自动退出。"
                      f"数据在 {self.out_path()}")
                self.quit = True
                return
        self.new_episode()

    def out_path(self) -> Path:
        return Path(self.args.out_dir) / "data.npz"

    def save(self):
        path = self.out_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 observations=np.stack(self.obs_list).astype(np.float32),
                 actions=np.stack(self.act_list).astype(np.float32),
                 episode_ends=np.array(self.ep_ends, dtype=np.int64))

    def load_existing(self):
        """把已有数据读进来接着录（默认追加，不会清掉之前的成果）。"""
        path = self.out_path()
        if self.args.overwrite or not path.exists():
            return
        d = np.load(path)
        obs, act, ends = d["observations"], d["actions"], d["episode_ends"]
        if act.ndim != 2 or act.shape[1] != 7:
            print(f"！{path} 里的 action 维度是 {act.shape}，和当前约定(action=7)不一致，"
                  f"为避免拼坏数据，本次不改动它。请换 --out_dir 或加 --overwrite。")
            self.quit = True
            return
        self.obs_list = list(obs)
        self.act_list = list(act)
        self.ep_ends = [int(x) for x in ends]
        self.saved_episodes = len(self.ep_ends)
        print(f"已有数据：{len(self.obs_list)} 帧 / {self.saved_episodes} 段 -> 继续追加到同一个文件")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self):
        env = self.env
        self.load_existing()
        if self.quit:
            return
        self.new_episode()
        print(KEYHELP)
        print(f"\n目标 {self.args.episodes} 段（含已有的 {self.saved_episodes} 段）；自动保存："
              f"{'开（放置完成后稳定 ' + format(self.args.auto_hold, '.1f') + 's 就自动存并换布局）' if self.args.auto_success else '关'}")
        print(f"输出：{self.out_path()}    速度 {self.speed:.2f} m/s"
              f"    录制：每 {env.model.opt.timestep * self.args.rec_every * 1000:.0f}ms 一帧")
        print("按 Enter 手动结束并存本段；Backspace 放弃；R 换布局；T 重来；Esc 退出\n", flush=True)

        # 自己开 OpenCV 窗口显示（不用 mujoco.viewer：它的可视化快捷键会和移动键撞车）
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, cam)
        cam.lookat[:] = [1.35, -0.02, 0.44]
        cam.distance = 1.75
        cam.azimuth = 118
        cam.elevation = -24
        view = TkView(env.model, self.args.width, self.args.height, cam)
        print("tkinter 窗口已打开（不是 MuJoCo 窗口）。鼠标左键拖动=转视角，右键=平移，滚轮=缩放。\n", flush=True)

        dt = self.args.ctrl_steps * env.model.opt.timestep
        try:
            while not self.quit:
                t0 = time.time()
                ev = view.poll_event()
                while ev is not None:       # 把这一拍排队的事件都处理掉
                    self.handle_event(ev[0], ev[1])
                    ev = view.poll_event()
                self.move_target(dt)
                self.control_tick()
                if self.ep_peaks:
                    self.ep_peaks.append(env.get_object_pos()[2])
                if self.message:
                    print(f"  [{self.message}]", flush=True)
                    self.message = ""
                self.maybe_autosave(time.time())
                if not view.alive:          # 用户直接关窗口 -> 优雅退出（别崩）
                    self.quit = True
                    break
                view.draw(env.data, self.hud())
                time.sleep(max(0.0, dt - (time.time() - t0)))
        except tk.TclError as e:
            print(f"  [窗口被关闭：{e}]", flush=True)
        finally:
            view.close()
            # 退出时兜底：没结算的那一段如果其实已经放好了，也存下来，别白录
            if self.stream and not self.ep_saved:
                if self.is_placed() and len(self.stream) >= self.args.min_frames:
                    print("\n退出时检测到本段已完成，自动保存。")
                    self.finish_episode(keep=True, auto=True)
                else:
                    print("\n退出：最后一段没结算（物块没放好），已丢弃。")
            if self.saved_episodes:
                print(f"共保存 {self.saved_episodes} 段 / {len(self.obs_list)} 帧 -> {self.out_path()}")
            else:
                print("没有保存任何数据。")

    def hud(self):
        """画在画面左上角的状态提示（用系统字体，可以写中文）。"""
        env = self.env
        tip = "放好了，正在自动保存…" if self.is_placed() else "把方块放到蓝色圆盘上"
        return [
            f"已保存 {self.saved_episodes}/{self.args.episodes} 段   累计 {len(self.obs_list)} 帧   "
            f"速度 {self.speed:.2f} m/s   夹爪 {'张开' if self.gripper_open else '合上'}",
            f"工具点 x={self.p_target[0]:.3f}  y={self.p_target[1]:.3f}  z={self.p_target[2]:.3f}    {tip}",
            "W/S 前后    A/D 左右    Q/E 升降    空格 夹爪     [ ] 调速",
            "T 重来(同布局)    R 换新布局    Enter 保存本段    Backspace 放弃    Esc 退出",
            "鼠标：左键拖动转视角，右键拖动平移，滚轮缩放",
        ]

    # ------------------------------------------------------------------
    # 无窗口自测：用"脚本化的假人"把目标沿抓取路径推一遍，验证录制与存盘
    # ------------------------------------------------------------------
    def selftest(self, episodes: int = 1):
        env = self.env
        place = None
        self.new_episode()
        place = None
        for ep in range(episodes):
            base = self.saved_episodes
            print(f"[selftest {ep + 1}/{episodes}] pick={np.round(self.ep_obj0[:2], 3)} "
                  f"place={np.round(self.ep_place[:2], 3)}")
            waypoints = [
                (self.ep_obj0 + [0, 0, 0.10], True),
                (self.ep_obj0, True),
                (self.ep_obj0, False),
                (self.ep_obj0 + [0, 0, 0.14], False),
                (self.ep_place + [0, 0, 0.10], False),
                (self.ep_place + [0, 0, 0.02], False),
                (self.ep_place + [0, 0, 0.02], True),
            ]
            place = self.ep_place.copy()
            for target, open_ in waypoints:
                self.gripper_open = open_
                target = np.asarray(target, dtype=float)
                for _ in range(400):
                    if np.linalg.norm(self.p_target - target) < 0.002:
                        break
                    d = target - self.p_target
                    n = np.linalg.norm(d)
                    self.p_target = self.p_target + d / n * min(0.005, n)
                    self.control_tick()
                    self.ep_peaks.append(env.get_object_pos()[2])
                    # 和交互模式走同一条自动保存判定（把"放置完成自动存"也测到）
                    if self.maybe_autosave(time.time()):
                        break
                if self.saved_episodes > base:      # 本段已经被自动保存 -> 结束
                    break
            # 没被自动保存的话，留一点真实时间给自动保存判定（它按墙上时间计时）
            t_end = time.time() + 3.0
            while self.saved_episodes == base and time.time() < t_end:
                self.control_tick()
                self.ep_peaks.append(env.get_object_pos()[2])
                self.maybe_autosave(time.time())
                time.sleep(0.02)
            if self.saved_episodes == base:
                self.finish_episode(keep=True)
            if self.saved_episodes >= self.args.episodes:
                print(f"（已录满 {self.args.episodes} 段，停止）")
                break
        self.ep_place = place
        # 回放最后一段录下来的数据，确认这套录制约定能复现任务（与 eval_grasp.py 的执行方式一致）
        self.replay_check()

    def replay_check(self):
        env = self.env
        ep = getattr(self, "last_episode", None)
        if not ep:
            print("[selftest] 没录到帧")
            return
        # 用同一组布局重放刚才录下来的动作
        obj_pose, place, frames = ep
        env.reset(object_pose=obj_pose)
        env.place_on_table(place)
        obs0 = env.get_object_pos().copy()
        peaks = [obs0[2]]
        seq = list(range(0, len(frames), self.args.lookahead))
        seq += [seq[-1]] * 12          # 末尾多跑几拍，让缓开爪那段走完
        for i in seq:
            act = frames[i][1]
            q = np.clip(act[:6], env.q_min, env.q_max)
            jaw = float(np.clip(act[6], env.jaw_min, env.jaw_max))
            env.data.ctrl[:] = np.concatenate([q, [jaw, jaw]])
            for _ in range(self.args.lookahead * self.args.ctrl_steps):
                mujoco.mj_step(env.model, env.data)
                peaks.append(env.get_object_pos()[2])
        err = float(np.linalg.norm(env.get_object_pos()[:2] - place[:2]))
        print(f"[selftest] 回放录制数据：抬升 {(max(peaks) - obs0[2]) * 1000:+.0f}mm  "
              f"放置水平误差 {err * 1000:.0f}mm  success={env.success()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/manual")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ctrl_steps", type=int, default=25,
                        help="每个控制周期的物理步数（25 步 * 2ms = 50ms -> 20Hz）")
    parser.add_argument("--rec_every", type=int, default=25, help="每多少个物理步录一帧")
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD,
                        help="动作取未来第几拍（要与专家数据的约定一致）")
    parser.add_argument("--speed", type=float, default=0.12, help="齿面中点移动速度 m/s")
    parser.add_argument("--hold", type=float, default=0.12,
                        help="松开按键后仍继续运动多久（防止松开事件丢失导致不动，别设太大）")
    parser.add_argument("--ik_iters", type=int, default=12, help="每拍做几步齿面中点 IK")
    parser.add_argument("--jaw_rate", type=float, default=0.10,
                        help="夹爪张开速率 rad/s（太快会弹飞物块）")
    parser.add_argument("--width", type=int, default=900, help="OpenCV 显示窗口宽度")
    parser.add_argument("--height", type=int, default=650, help="OpenCV 显示窗口高度")
    parser.add_argument("--open_margin", type=float, default=0.030,
                        help="遥操作时夹爪张开的富余量(m)：净距 = 物块边长 + 这个值。"
                             "实测权衡：14mm -> 每边余量 7mm、松手误差 8mm；"
                             "36mm -> 每边 18mm、松手误差 43mm（逼近 50mm 判定线）。"
                             "30mm -> 每边约 15mm，够人手动对位")
    parser.add_argument("--episodes", type=int, default=20,
                        help="要录多少段（录满自动退出；selftest 模式同义）")
    parser.add_argument("--auto_success", dest="auto_success", action="store_true", default=True,
                        help="物块放好后自动结算并换布局（默认开）")
    parser.add_argument("--no_auto_success", dest="auto_success", action="store_false",
                        help="关掉自动保存，全部手动按 Enter 结算")
    parser.add_argument("--auto_hold", type=float, default=0.4,
                        help="放置完成要连续成立多久才自动保存（秒）")
    parser.add_argument("--min_frames", type=int, default=20,
                        help="太短的段不予自动保存（防止一开局就误触发）")
    parser.add_argument("--keep_failed", action="store_true",
                        help="允许把没完成的段也写进数据集（默认只存成功的示范）")
    parser.add_argument("--overwrite", action="store_true",
                        help="忽略已有 data.npz，从零开始（默认是追加）")
    parser.add_argument("--selftest", action="store_true",
                        help="不开窗口，用脚本化的假人跑一遍验证录制/存盘")
    args = parser.parse_args()

    if args.selftest and args.out_dir == "data/manual":
        # 自测绝对不能往正式目录写（曾经把录好的数据覆盖掉过）
        args.out_dir = "data/manual_selftest"
        print(f"[selftest] 输出改到 {args.out_dir}，不会碰 data/manual 里的正式数据")

    env = UranusGraspEnv(seed=args.seed)
    col = TeleopCollector(env, args)
    if args.selftest:
        col.selftest(args.episodes)
    else:
        col.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
