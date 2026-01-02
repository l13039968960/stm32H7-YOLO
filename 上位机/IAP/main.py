import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import serial
import serial.tools.list_ports
import threading
import queue
import time
from datetime import datetime
import json
import os
import re
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# 设置Matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体，如果不存在则使用DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class FaceDetectionMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("STM32人脸检测监控系统")
        self.root.geometry("1200x700")

        # 串口相关变量
        self.serial_port = None
        self.is_connected = False
        self.receive_thread = None
        self.stop_receive = False
        self.data_queue = queue.Queue()

        # 人脸检测数据
        self.current_frame = 0
        self.face_count_history = deque(maxlen=50)  # 保存最近50帧的人脸数量
        self.frame_history = deque(maxlen=50)  # 保存最近50帧的编号
        self.detected_faces = []  # 当前帧检测到的人脸
        self.total_faces_detected = 0  # 总检测到的人脸数
        self.current_face_count = 0  # 当前帧的人脸数量
        self.frame_buffer = ""  # 用于缓冲不完整的帧数据

        # 图像尺寸设置
        self.image_width = 112  # 根据您的描述，图像是112x112像素
        self.image_height = 112

        # 保存设置的文件
        self.config_file = "face_detection_config.json"

        # 创建UI
        self.create_widgets()
        self.load_config()

        # 启动数据更新线程
        self.update_display()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 串口配置区域
        config_frame = ttk.LabelFrame(main_frame, text="串口配置", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 串口号
        ttk.Label(config_frame, text="串口号:").grid(row=0, column=0, sticky=tk.W)
        self.port_combo = ttk.Combobox(config_frame, width=15)
        self.port_combo.grid(row=0, column=1, padx=(5, 20))
        self.refresh_ports()

        # 波特率
        ttk.Label(config_frame, text="波特率:").grid(row=0, column=2, sticky=tk.W)
        self.baudrate_combo = ttk.Combobox(config_frame, width=10,
                                           values=['9600', '19200', '38400', '57600', '115200', '230400', '460800',
                                                   '921600'])
        self.baudrate_combo.set('115200')
        self.baudrate_combo.grid(row=0, column=3, padx=5)

        # 控制按钮
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=0, column=4, padx=(20, 0))

        self.connect_btn = ttk.Button(button_frame, text="打开串口", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=0, padx=2)

        self.refresh_btn = ttk.Button(button_frame, text="刷新串口", command=self.refresh_ports)
        self.refresh_btn.grid(row=0, column=1, padx=2)

        # 统计信息区域
        stats_frame = ttk.LabelFrame(config_frame, text="统计信息", padding="10")
        stats_frame.grid(row=0, column=5, padx=(40, 0))

        self.frame_label = ttk.Label(stats_frame, text="帧号: 0")
        self.frame_label.grid(row=0, column=0, padx=5)

        self.face_count_label = ttk.Label(stats_frame, text="检测人脸: 0")
        self.face_count_label.grid(row=0, column=1, padx=5)

        self.total_faces_label = ttk.Label(stats_frame, text="总检测: 0")
        self.total_faces_label.grid(row=0, column=2, padx=5)

        # 主显示区域 - 使用PanedWindow实现可调节分割
        paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashwidth=10)
        paned.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 左侧：串口数据显示（缩小版）
        left_frame = ttk.LabelFrame(paned, text="串口原始数据", padding="10")

        # 接收文本框（缩小高度）
        self.receive_text = scrolledtext.ScrolledText(left_frame, width=50, height=15, wrap=tk.WORD)
        self.receive_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 显示选项
        option_frame = ttk.Frame(left_frame)
        option_frame.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))

        self.auto_scroll = tk.BooleanVar(value=True)
        scroll_check = ttk.Checkbutton(option_frame, text="自动滚动", variable=self.auto_scroll)
        scroll_check.grid(row=0, column=0, padx=(0, 20))

        self.show_raw_data = tk.BooleanVar(value=True)
        raw_check = ttk.Checkbutton(option_frame, text="显示原始数据", variable=self.show_raw_data)
        raw_check.grid(row=0, column=1, padx=(0, 20))

        clear_btn = ttk.Button(option_frame, text="清空数据", command=self.clear_receive)
        clear_btn.grid(row=0, column=2, padx=2)

        save_btn = ttk.Button(option_frame, text="保存数据", command=self.save_data)
        save_btn.grid(row=0, column=3, padx=2)

        # 右侧：人脸检测信息（合并到一个页面）
        right_frame = ttk.LabelFrame(paned, text="人脸检测信息", padding="10")

        # 创建垂直分割的PanedWindow来组织右侧内容
        right_paned = tk.PanedWindow(right_frame, orient=tk.VERTICAL, sashwidth=10)
        right_paned.pack(fill=tk.BOTH, expand=True)

        # 上部：人脸列表
        face_list_frame = ttk.LabelFrame(right_paned, text="人脸列表", padding="5")

        # 创建Treeview显示人脸信息
        columns = ('ID', 'X1', 'Y1', 'X2', 'Y2', '置信度')
        self.face_tree = ttk.Treeview(face_list_frame, columns=columns, show='headings', height=6)

        # 设置列标题
        for col in columns:
            self.face_tree.heading(col, text=col)
            self.face_tree.column(col, width=70)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(face_list_frame, orient=tk.VERTICAL, command=self.face_tree.yview)
        self.face_tree.configure(yscrollcommand=scrollbar.set)

        self.face_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        right_paned.add(face_list_frame)

        # 中部：统计图表
        chart_frame = ttk.LabelFrame(right_paned, text="统计图表", padding="5")

        # 创建Matplotlib图表
        self.fig = Figure(figsize=(6, 3), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('人脸检测统计')
        self.ax.set_xlabel('帧号')
        self.ax.set_ylabel('人脸数量')
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right_paned.add(chart_frame)

        # 下部：模拟显示
        simulation_frame = ttk.LabelFrame(right_paned, text="模拟显示", padding="5")

        # 创建模拟画布
        self.sim_canvas = tk.Canvas(simulation_frame, width=400, height=250, bg='black',
                                    highlightthickness=1, highlightbackground="gray")
        self.sim_canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # 添加模拟图像标签
        self.sim_label = ttk.Label(simulation_frame, text=f"图像尺寸: {self.image_width}x{self.image_height}")
        self.sim_label.pack()

        right_paned.add(simulation_frame)

        # 设置右侧区域各部分比例
        right_paned.sash_place(0, 0, 150)  # 人脸列表区域高度
        right_paned.sash_place(1, 0, 300)  # 统计图表区域高度

        # 将左右框架添加到主PanedWindow
        paned.add(left_frame)
        paned.add(right_frame)

        # 配置权重
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        # 状态栏
        self.status_bar = ttk.Label(main_frame, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

    def refresh_ports(self):
        """刷新可用串口列表"""
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        self.port_combo['values'] = port_list
        if port_list:
            self.port_combo.set(port_list[0])

    def toggle_connection(self):
        """打开/关闭串口连接"""
        if not self.is_connected:
            self.connect_serial()
        else:
            self.disconnect_serial()

    def connect_serial(self):
        """打开串口连接"""
        port = self.port_combo.get()
        if not port:
            messagebox.showerror("错误", "请选择串口号")
            return

        try:
            # 获取串口参数
            baudrate = int(self.baudrate_combo.get())

            # 打开串口
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=8,
                parity=serial.PARITY_NONE,
                stopbits=1,
                timeout=1
            )

            self.is_connected = True
            self.stop_receive = False

            # 启动接收线程
            self.receive_thread = threading.Thread(target=self.receive_data, daemon=True)
            self.receive_thread.start()

            # 更新UI
            self.connect_btn.config(text="关闭串口")
            self.status_bar.config(text=f"已连接 {port} @ {baudrate}bps")

            # 保存配置
            self.save_config()

        except Exception as e:
            messagebox.showerror("连接错误", f"无法打开串口:\n{str(e)}")

    def disconnect_serial(self):
        """关闭串口连接"""
        if self.serial_port and self.serial_port.is_open:
            self.stop_receive = True
            time.sleep(0.1)  # 给线程一点时间退出

            try:
                self.serial_port.close()
            except:
                pass

        self.is_connected = False
        self.connect_btn.config(text="打开串口")
        self.status_bar.config(text="已断开连接")

    def receive_data(self):
        """接收串口数据的线程函数"""
        buffer = ""
        while not self.stop_receive and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data:
                        try:
                            text = data.decode('utf-8', errors='replace')
                            buffer += text

                            # 按行分割处理
                            lines = buffer.split('\n')
                            # 保留最后不完整的行
                            buffer = lines[-1]

                            # 处理完整的行
                            for line in lines[:-1]:
                                line = line.strip()
                                if line:
                                    self.data_queue.put(line)
                        except:
                            pass
            except Exception as e:
                if not self.stop_receive:
                    # 修复lambda作用域问题
                    error_msg = str(e)
                    self.root.after(0, lambda err=error_msg: self.show_receive_error(err))
                break
            time.sleep(0.01)

    def show_receive_error(self, error_msg):
        """显示接收错误信息"""
        messagebox.showerror("接收错误", f"接收数据时出错:\n{error_msg}")

    def parse_frame_data(self, data_lines):
        """解析完整的一帧数据"""
        faces = []
        frame_num = 0
        face_count = 0

        for line in data_lines:
            # 解析帧号
            frame_match = re.search(r'=== Frame (\d+) ===', line)
            if frame_match:
                try:
                    frame_num = int(frame_match.group(1))
                except:
                    frame_num = 0

            # 解析人脸信息 - 注意：现在是x1,y1,x2,y2格式
            face_match = re.search(
                r'\[Face\s+(\d+)\]\s+BBox:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Conf:\s*([\d\.]+)', line)
            if face_match:
                try:
                    face_id = int(face_match.group(1))
                    x1 = int(face_match.group(2))
                    y1 = int(face_match.group(3))
                    x2 = int(face_match.group(4))
                    y2 = int(face_match.group(5))
                    conf = float(face_match.group(6))

                    # 计算宽度和高度
                    width = x2 - x1
                    height = y2 - y1

                    faces.append({
                        'id': face_id,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'width': width,
                        'height': height,
                        'confidence': conf
                    })
                except Exception as e:
                    print(f"解析人脸信息出错: {e}")

            # 解析总人脸数
            total_match = re.search(r'Total faces detected:\s*(\d+)', line, re.IGNORECASE)
            if total_match:
                try:
                    face_count = int(total_match.group(1))
                except:
                    face_count = len(faces)  # 如果解析失败，使用实际检测到的人脸数

        return frame_num, faces, face_count

    def process_received_data(self, line):
        """处理接收到的单行数据"""
        # 在接收文本框中显示
        if self.show_raw_data.get():
            timestamp = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
            self.receive_text.insert(tk.END, timestamp + line + '\n')

        # 添加到帧缓冲区
        self.frame_buffer += line + "\n"

        # 检查是否收到完整的一帧（以Total faces detected结尾）
        if "Total faces detected:" in line or "Total faces detected:" in line.upper():
            # 解析完整帧数据
            frame_lines = self.frame_buffer.strip().split('\n')
            frame_num, faces, face_count = self.parse_frame_data(frame_lines)

            # 更新数据
            self.current_frame = frame_num
            self.detected_faces = faces
            self.current_face_count = face_count

            # 更新历史数据
            if len(self.frame_history) == 0 or frame_num != self.frame_history[-1]:
                self.frame_history.append(frame_num)
                self.face_count_history.append(face_count)
                self.total_faces_detected += face_count

            # 清空缓冲区
            self.frame_buffer = ""

        # 自动滚动
        if self.auto_scroll.get():
            self.receive_text.see(tk.END)

    def update_display(self):
        """更新显示内容"""
        try:
            # 处理所有排队的数据
            while not self.data_queue.empty():
                line = self.data_queue.get_nowait()
                self.process_received_data(line)

        except queue.Empty:
            pass

        # 更新UI显示
        self.update_face_display()
        self.update_stats_display()
        self.update_chart()
        self.update_simulation()

        # 每隔50ms再次调用（提高更新频率）
        self.root.after(50, self.update_display)

    def update_face_display(self):
        """更新人脸列表显示"""
        # 清空现有内容
        for item in self.face_tree.get_children():
            self.face_tree.delete(item)

        # 添加新人脸数据
        for face in self.detected_faces:
            self.face_tree.insert('', 'end', values=(
                face['id'],
                face['x1'],
                face['y1'],
                face['x2'],
                face['y2'],
                f"{face['confidence']:.3f}"
            ))

    def update_stats_display(self):
        """更新统计信息显示"""
        self.frame_label.config(text=f"帧号: {self.current_frame}")
        self.face_count_label.config(text=f"检测人脸: {self.current_face_count}")
        self.total_faces_label.config(text=f"总检测: {self.total_faces_detected}")

    def update_chart(self):
        """更新统计图表"""
        if len(self.frame_history) > 0 and len(self.face_count_history) > 0:
            try:
                self.ax.clear()

                # 绘制折线图
                self.ax.plot(list(self.frame_history), list(self.face_count_history),
                             'b-', linewidth=2, marker='o', markersize=4)

                # 设置图表属性
                self.ax.set_title('人脸检测统计')
                self.ax.set_xlabel('帧号')
                self.ax.set_ylabel('人脸数量')
                self.ax.grid(True, alpha=0.3)

                # 设置Y轴范围
                if len(self.face_count_history) > 0:
                    max_val = max(self.face_count_history)
                    self.ax.set_ylim(0, max(5, max_val + 1))

                # 重新绘制
                self.canvas.draw()
            except Exception as e:
                print(f"更新图表出错: {e}")

    def update_simulation(self):
        """更新模拟显示 - 使用x1,y1,x2,y2格式"""
        try:
            self.sim_canvas.delete("all")

            # 绘制模拟图像边界
            padding = 20
            canvas_width = self.sim_canvas.winfo_width() or 400
            canvas_height = self.sim_canvas.winfo_height() or 250

            # 计算缩放比例，保持图像宽高比
            scale_x = (canvas_width - 2 * padding) / self.image_width
            scale_y = (canvas_height - 2 * padding) / self.image_height
            scale = min(scale_x, scale_y)  # 使用较小的比例保持宽高比

            # 计算居中位置
            display_width = self.image_width * scale
            display_height = self.image_height * scale
            x_offset = (canvas_width - display_width) / 2
            y_offset = (canvas_height - display_height) / 2

            # 绘制图像区域
            self.sim_canvas.create_rectangle(
                x_offset, y_offset,
                x_offset + display_width, y_offset + display_height,
                outline="white", width=2
            )

            # 绘制人脸框（使用x1,y1,x2,y2格式）
            for face in self.detected_faces:
                # 计算人脸框在画布上的坐标
                x1 = x_offset + face['x1'] * scale
                y1 = y_offset + face['y1'] * scale
                x2 = x_offset + face['x2'] * scale
                y2 = y_offset + face['y2'] * scale

                # 确保坐标在合理范围内
                x1 = max(x_offset, min(x1, x_offset + display_width))
                y1 = max(y_offset, min(y1, y_offset + display_height))
                x2 = max(x_offset, min(x2, x_offset + display_width))
                y2 = max(y_offset, min(y2, y_offset + display_height))

                # 绘制人脸框
                self.sim_canvas.create_rectangle(x1, y1, x2, y2,
                                                 outline="red", width=2)

                # 绘制置信度文本
                conf_text = f"{face['confidence']:.2f}"
                text_x = x1
                text_y = max(y_offset, y1 - 15)  # 确保文本不会超出画布顶部

                # 绘制半透明背景提高文本可读性
                self.sim_canvas.create_rectangle(
                    text_x, text_y, text_x + len(conf_text) * 6, text_y + 15,
                    fill="black", outline="black"
                )

                self.sim_canvas.create_text(text_x + 2, text_y + 2, text=conf_text,
                                            fill="yellow", anchor="nw", font=("Arial", 8))

                # 绘制人脸ID
                id_text = f"Face {face['id']}"
                id_y = min(y_offset + display_height - 15, y2 + 2)  # 确保ID不会超出画布底部

                # 绘制半透明背景
                self.sim_canvas.create_rectangle(
                    x1, id_y, x1 + len(id_text) * 6, id_y + 15,
                    fill="black", outline="black"
                )

                self.sim_canvas.create_text(x1 + 2, id_y + 2, text=id_text,
                                            fill="white", anchor="nw", font=("Arial", 8))

            # 更新标签显示当前帧信息
            self.sim_label.config(
                text=f"图像尺寸: {self.image_width}x{self.image_height} | 检测到: {len(self.detected_faces)}人 | 帧号: {self.current_frame}")

        except Exception as e:
            print(f"更新模拟显示出错: {e}")

    def clear_receive(self):
        """清空接收区"""
        self.receive_text.delete(1.0, tk.END)
        # 同时重置统计信息
        self.current_frame = 0
        self.face_count_history.clear()
        self.frame_history.clear()
        self.detected_faces.clear()
        self.total_faces_detected = 0
        self.current_face_count = 0
        self.frame_buffer = ""
        self.update_face_display()
        self.update_stats_display()
        self.update_chart()
        self.update_simulation()

    def save_data(self):
        """保存接收的数据到文件"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.receive_text.get(1.0, tk.END))
                self.status_bar.config(text=f"数据已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存文件时出错:\n{str(e)}")

    def save_config(self):
        """保存串口配置"""
        config = {
            'port': self.port_combo.get(),
            'baudrate': self.baudrate_combo.get(),
            'image_width': self.image_width,
            'image_height': self.image_height
        }

        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except:
            pass

    def load_config(self):
        """加载串口配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                if config.get('port') in self.port_combo['values']:
                    self.port_combo.set(config.get('port', ''))
                self.baudrate_combo.set(config.get('baudrate', '115200'))
                self.image_width = config.get('image_width', 112)
                self.image_height = config.get('image_height', 112)
            except:
                pass

    def on_closing(self):
        """窗口关闭时的清理"""
        self.disconnect_serial()
        self.save_config()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceDetectionMonitor(root)

    # 设置关闭窗口时的处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()


if __name__ == "__main__":
    main()