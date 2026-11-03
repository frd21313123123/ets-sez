"""The launcher window: pick a date, see the snow it brings, press Play."""
import calendar
import datetime as dt
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from . import build, model, profiles, session
from .paths import ROOT, default_paths

BG = "#14181d"
PANEL = "#1c2229"
RAISED = "#262e37"
TEXT = "#e8ebef"
MUTED = "#8d97a3"
ACCENT = "#5aa9e6"
ACCENT_DARK = "#3f8bc7"
ON_ACCENT = "#0d1216"
WARN = "#e0a84a"
ERROR = "#e5736b"
LAND = np.array((58, 80, 56))
SNOW = np.array((238, 243, 250))

FONT = "Segoe UI"
MONTHS = ("Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август", "Сентябрь", "Октябрь",
          "Ноябрь", "Декабрь")
MONTHS_OF = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
             "ноября", "декабря")
WEEKDAYS = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
SEASON_PRESETS = (("Зима", 1, 20), ("Весна", 4, 5), ("Лето", 7, 15), ("Осень", 10, 15))
TEMPERATURE_CITIES = (("oslo", "Осло"), ("berlin", "Берлин"), ("madrid", "Мадрид"))
SNOW_VISIBLE_FROM = 1 / (2 * model.SNOW_LEVELS)  # the cover at which the first snow level shows
NEW_PROFILE_NOTE = " (будет создан)"


def plural_percent(date):
    return f"{date.day} {MONTHS_OF[date.month - 1]} {date.year}"


def signed(value):
    rounded = int(round(value))
    return f"+{rounded}°" if rounded > 0 else f"−{abs(rounded)}°" if rounded < 0 else "0°"


class App(tk.Tk):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.table = model.ZoneTable()
        self.state_data = session.load_state(paths)
        self.events = queue.Queue()
        self.busy = False
        self.map_photo = None

        saved = self._load_settings()
        self.custom_date = self._saved_date(saved)
        self.use_today = tk.BooleanVar(value=saved.get("mode", "today") == "today")
        shown = dt.date.today() if self.use_today.get() else self.custom_date
        self.view_year, self.view_month = shown.year, shown.month
        self.profile_var = tk.StringVar()
        self.saved_profile = saved.get("profile", profiles.DEFAULT_PROFILE)

        self.title("Сезоны ETS 2")
        self.configure(bg=BG)
        self.geometry("980x640")
        self.minsize(900, 600)
        self._style()
        self._layout()
        self._reload_profiles()
        self.refresh()
        self.after(100, self._pump)
        self._ensure_map_data()

    # Settings

    def _load_settings(self):
        try:
            with open(self.paths.settings, encoding="utf8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _saved_date(self, saved):
        try:
            return dt.date.fromisoformat(saved.get("date", ""))
        except ValueError:
            return dt.date.today()

    def _save_settings(self):
        data = {"mode": "today" if self.use_today.get() else "custom", "date": self.custom_date.isoformat(),
                "profile": self.selected_profile_name()}
        os.makedirs(self.paths.out, exist_ok=True)
        with open(self.paths.settings, "w", encoding="utf8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    def play_date(self):
        return dt.date.today() if self.use_today.get() else self.custom_date

    # Layout

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=RAISED, background=RAISED, foreground=TEXT, arrowcolor=TEXT,
                        bordercolor=RAISED, lightcolor=RAISED, darkcolor=RAISED, padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", RAISED)], foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", RAISED)], selectforeground=[("readonly", TEXT)])
        self.option_add("*TCombobox*Listbox.background", RAISED)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", ON_ACCENT)
        self.option_add("*TCombobox*Listbox.font", (FONT, 10))

    def _label(self, parent, text="", size=10, color=TEXT, bold=False, bg=PANEL, **kw):
        return tk.Label(parent, text=text, bg=bg, fg=color, font=(FONT, size, "bold" if bold else "normal"), **kw)

    def _button(self, parent, text, command, bg=RAISED, fg=TEXT, size=10, bold=False, **kw):
        active = ACCENT_DARK if bg == ACCENT else "#323c47"
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=active,
                         activeforeground=fg, relief="flat", bd=0, cursor="hand2",
                         font=(FONT, size, "bold" if bold else "normal"), **kw)

    def _layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = tk.Frame(self, bg=PANEL, padx=18, pady=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=(14, 7))
        self._label(left, "Дата игры", 13, bold=True).pack(anchor="w")

        modes = tk.Frame(left, bg=PANEL)
        modes.pack(fill="x", pady=(10, 12))
        modes.columnconfigure((0, 1), weight=1)
        self.today_btn = self._button(modes, "Как сегодня", lambda: self.set_mode(True), pady=6)
        self.custom_btn = self._button(modes, "Своя дата", lambda: self.set_mode(False), pady=6)
        self.today_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.custom_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        head = tk.Frame(left, bg=PANEL)
        head.pack(fill="x")
        self._button(head, "‹", lambda: self.shift_month(-1), width=3, size=12).pack(side="left")
        self.month_label = self._label(head, "", 11, bold=True)
        self.month_label.pack(side="left", expand=True)
        self._button(head, "›", lambda: self.shift_month(1), width=3, size=12).pack(side="right")

        self.grid_frame = tk.Frame(left, bg=PANEL)
        self.grid_frame.pack(pady=(10, 12))

        presets = tk.Frame(left, bg=PANEL)
        presets.pack(fill="x")
        presets.columnconfigure((0, 1, 2, 3), weight=1)
        for col, (title, month, day) in enumerate(SEASON_PRESETS):
            self._button(presets, title, lambda m=month, d=day: self.pick_preset(m, d), pady=5).grid(
                row=0, column=col, sticky="ew", padx=2)

        right = tk.Frame(self, bg=PANEL, padx=18, pady=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 14), pady=(14, 7))
        self.date_title = self._label(right, "", 13, bold=True)
        self.date_title.pack(anchor="w")
        self.date_info = self._label(right, "", 10, MUTED, justify="left")
        self.date_info.pack(anchor="w", pady=(2, 8))
        self.map_label = tk.Label(right, bg=PANEL)
        self.map_label.pack(fill="both", expand=True)
        self.map_label.bind("<Configure>", lambda _e: self.draw_map())

        bottom = tk.Frame(self, bg=PANEL, padx=18, pady=14)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(7, 14))
        bottom.columnconfigure(1, weight=1)
        self._label(bottom, "Профиль").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.profile_box = ttk.Combobox(bottom, textvariable=self.profile_var, state="readonly", width=34,
                                        font=(FONT, 10))
        self.profile_box.grid(row=0, column=1, sticky="w")
        self.profile_box.bind("<<ComboboxSelected>>", lambda _e: self.refresh_profile())
        self.play_btn = self._button(bottom, "Играть", self.play, bg=ACCENT, fg=ON_ACCENT, size=13, bold=True,
                                     padx=38, pady=8)
        self.play_btn.grid(row=0, column=2, rowspan=3, sticky="e", padx=(14, 0))
        self.rebuild_btn = self._button(bottom, "Пересобрать карту", self.rebuild, padx=12, pady=8)
        self.mods_label = self._label(bottom, "", 9, MUTED, justify="left", anchor="w")
        self.mods_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.status = self._label(bottom, "", 10, MUTED, justify="left", anchor="w")
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.bind("<Configure>", self._wrap_text)

    def _wrap_text(self, _event=None):
        width = max(300, self.winfo_width() - 300)
        self.mods_label.configure(wraplength=width)
        self.status.configure(wraplength=width)

    # Date

    def set_mode(self, today):
        self.use_today.set(today)
        date = self.play_date()
        self.view_year, self.view_month = date.year, date.month
        self.refresh()

    def shift_month(self, delta):
        index = self.view_year * 12 + self.view_month - 1 + delta
        self.view_year, self.view_month = index // 12, index % 12 + 1
        self.draw_calendar()

    def pick_day(self, day):
        self.custom_date = dt.date(self.view_year, self.view_month, day)
        self.use_today.set(False)
        self.refresh()

    def pick_preset(self, month, day):
        self.custom_date = dt.date(self.play_date().year, month, day)
        self.use_today.set(False)
        self.view_year, self.view_month = self.custom_date.year, month
        self.refresh()

    def draw_calendar(self):
        self.month_label.configure(text=f"{MONTHS[self.view_month - 1]} {self.view_year}")
        for child in self.grid_frame.winfo_children():
            child.destroy()
        for col, name in enumerate(WEEKDAYS):
            self._label(self.grid_frame, name, 9, MUTED, width=4).grid(row=0, column=col, pady=(0, 4))
        selected = self.play_date()
        today = dt.date.today()
        for row, week in enumerate(calendar.monthcalendar(self.view_year, self.view_month), start=1):
            for col, day in enumerate(week):
                if not day:
                    continue
                date = dt.date(self.view_year, self.view_month, day)
                chosen = date == selected
                fg = ON_ACCENT if chosen else ACCENT if date == today else TEXT
                self._button(self.grid_frame, str(day), lambda d=day: self.pick_day(d), width=4, pady=3,
                             bg=ACCENT if chosen else PANEL, fg=fg, bold=chosen or date == today).grid(
                    row=row, column=col, padx=1, pady=1)

    def refresh(self):
        today_mode = self.use_today.get()
        for button, on in ((self.today_btn, today_mode), (self.custom_btn, not today_mode)):
            button.configure(bg=ACCENT if on else RAISED, fg=ON_ACCENT if on else TEXT)
        self.draw_calendar()
        self.draw_map()
        self.refresh_rebuild()

    # Map

    def draw_map(self):
        date = self.play_date()
        day = model.day_of_year(date)
        prefix = "Сегодня, " if self.use_today.get() else ""
        self.date_title.configure(text=prefix + plural_percent(date))
        state = self.state_data or {}
        sectors = state.get("sectors")
        if not sectors:
            self.date_info.configure(text=f"День {day} в году")
            self.map_label.configure(image="", text="Карта появится после чтения данных игры", fg=MUTED,
                                     font=(FONT, 10))
            return
        covers = build.cover_map(sectors, self.table, day)
        shown = sum(c >= SNOW_VISIBLE_FROM for c in covers.values())
        located = state.get("located", {})
        parts = []
        for key, title in TEMPERATURE_CITIES:
            if key in located:
                jan, jul = model.STATIONS[key]
                parts.append(f"{title} {signed(model.temperature(jan, jul, min(day, 365)))}")
        self.date_info.configure(text=f"День {day} в году · снег на {100 * shown // len(covers)}% секторов · "
                                      + " · ".join(parts))
        self.map_photo = ImageTk.PhotoImage(self._render_map(covers))
        self.map_label.configure(image=self.map_photo, text="")

    def _render_map(self, covers):
        xs = [k[0] for k in covers]
        zs = [k[1] for k in covers]
        x0, z0 = min(xs), min(zs)
        nx, nz = max(xs) - x0 + 1, max(zs) - z0 + 1
        panel = np.array(self.winfo_rgb(PANEL)) // 256
        pixels = np.tile(panel.astype(np.uint8), (nz, nx, 1))
        for (x, z), cover in covers.items():
            pixels[z - z0, x - x0] = LAND + (SNOW - LAND) * cover
        width = max(self.map_label.winfo_width(), 200)
        height = max(self.map_label.winfo_height(), 200)
        scale = max(1, min(width // nx, height // nz))
        return Image.fromarray(pixels).resize((nx * scale, nz * scale), Image.NEAREST)

    # Profile

    def _reload_profiles(self):
        self.known_profiles = {p.name: p for p in profiles.list_profiles(self.paths.profiles)}
        names = sorted(self.known_profiles, key=str.lower)
        if profiles.DEFAULT_PROFILE in names:
            names.remove(profiles.DEFAULT_PROFILE)
            names.insert(0, profiles.DEFAULT_PROFILE)
        else:
            names.insert(0, profiles.DEFAULT_PROFILE + NEW_PROFILE_NOTE)
        self.profile_box.configure(values=names)
        wanted = self.saved_profile if self.saved_profile in names else names[0]
        self.profile_var.set(wanted)
        self.refresh_profile()

    def selected_profile_name(self):
        return self.profile_var.get().replace(NEW_PROFILE_NOTE, "")

    def refresh_profile(self):
        profile = self.known_profiles.get(self.selected_profile_name())
        foreign = profile.foreign_mods if profile else []
        order = " → ".join(title for _pkg, title in reversed(profiles.SEASON_MODS))
        if foreign:
            self.mods_label.configure(
                text=f"Порядок сверху вниз: {order}, затем ещё модов: {len(foreign)}. Сторонние карты и моды на "
                     "погоду с сезонами несовместимы, лучше взять отдельный профиль.", fg=WARN)
        else:
            self.mods_label.configure(text=f"Порядок сверху вниз: {order}. Других модов в профиле нет.", fg=MUTED)

    # Background work

    def run_bg(self, work, done):
        def worker():
            try:
                result = work()
            except Exception as exc:  # shown to the player rather than lost with the thread
                self.events.put(("error", exc))
            else:
                self.events.put(("done", (done, result)))

        self.set_busy(True)
        threading.Thread(target=worker, daemon=True).start()

    def post_status(self, text):
        self.events.put(("status", text))

    def _pump(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.show_status(value)
                elif kind == "error":
                    self.set_busy(False)
                    self.show_error(value)
                else:
                    self.set_busy(False)
                    value[0](value[1])
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.play_btn.configure(state=state, bg=RAISED if busy else ACCENT)
        self.rebuild_btn.configure(state=state)

    def show_status(self, text, color=MUTED):
        self.status.configure(text=text, fg=color)

    def show_error(self, exc):
        if isinstance(exc, session.GameRunning):
            self.show_status("Игра уже запущена. Закрой её: моды подключаются только при запуске.", WARN)
        elif isinstance(exc, session.NotBuilt):
            self.show_status("Мод с картой ещё не собран. Нажми «Пересобрать карту».", WARN)
        else:
            self.show_status(f"Ошибка: {exc}", ERROR)

    # Actions

    def _ensure_map_data(self):
        state = self.state_data
        if state is None or ("sectors" in state and "game_stamp" in state):
            return
        self.run_bg(lambda: session.complete_state(self.paths, state, self.post_status), self._map_data_ready)

    def _map_data_ready(self, state):
        self.state_data = state
        self.show_status("")
        self.refresh()

    def refresh_rebuild(self):
        state = self.state_data
        needed = state is None or session.game_updated(self.paths, state)
        if needed:
            self.rebuild_btn.grid(row=0, column=1, sticky="e")
            if state is None:
                self.show_status("Мод с картой ещё не собран. Нажми «Пересобрать карту».", WARN)
            else:
                self.show_status("Игра обновилась, мод с картой устарел. Пересобери его, иначе возможны сбои.", WARN)
        else:
            self.rebuild_btn.grid_remove()

    def play(self):
        if self.busy:
            return
        self._save_settings()
        date, name = self.play_date(), self.selected_profile_name()
        self.run_bg(lambda: session.prepare(self.paths, self.state_data, date, name, self.post_status),
                    lambda profile: self._launch(profile, date))

    def _launch(self, profile, date):
        self._reload_profiles()
        self.show_status(f"Игра запускается на {plural_percent(date)}. В игре выбери профиль «{profile.name}».")
        self.iconify()
        self.update()
        session.start_game()
        threading.Thread(target=session.bring_game_up, daemon=True).start()

    def rebuild(self):
        if self.busy:
            return
        if not os.path.isdir(self.paths.extracted):
            self.show_status("Нет распакованных текстур. Запусти tools\\extract_textures.bat и повтори.", WARN)
            return

        def work():
            process = subprocess.Popen(
                [sys.executable, "-m", "seasons", "map", "--textures", self.paths.extracted], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW)
            for line in process.stdout:
                self.post_status("Сборка: " + line.strip())
            if process.wait() != 0:
                raise RuntimeError("сборка карты не удалась")
            return session.load_state(self.paths)

        self.run_bg(work, self._map_data_ready)


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    App(default_paths()).mainloop()


if __name__ == "__main__":
    main()
