import openvino_genai as ov_genai
import time
import tkinter as tk
from tkinter import ttk, messagebox

model_list=['TinyLlama-1.1B']
device_list=['CPU','GPU','NPU']
m,n=0,0
pipe = None  # 确保在程序启动时 pipe 被定义

# GUI 主程序
def start_gui():
    def toggle_buttons(is_loaded):
        if is_loaded:
            load_button.grid_remove()
            unload_button.grid()
            model_menu.config(state=tk.DISABLED)  # 禁用模型选项卡
            device_menu.config(state=tk.DISABLED)  # 禁用设备选项卡
        else:
            unload_button.grid_remove()
            load_button.grid()
            model_menu.config(state=tk.NORMAL)  # 启用模型选项卡
            device_menu.config(state=tk.NORMAL)  # 启用设备选项卡

    def load_model():
        selected_model = model_var.get()
        selected_device = device_var.get()
        console_display.insert(tk.END, 'Loading......\n')
        console_display.see(tk.END)  # 确保最新消息可见
        root.update()  # 强制更新界面，确保立即显示 'Loading......'
        try:
            start_time = time.time()
            if selected_device == 'NPU':
                global pipe
                pipe = ov_genai.LLMPipeline(selected_model, selected_device, NPUW_CACHE_DIR=f".npucache/{selected_model}")
            else:
                pipe = ov_genai.LLMPipeline(selected_model, selected_device)
            end_time = time.time()
            load_time = end_time - start_time
            console_display.insert(tk.END, f"模型加载成功！耗时：{load_time:.2f} 秒\n\n")
            console_display.see(tk.END)
            toggle_buttons(True)
            update_send_button()
        except Exception as e:
            console_display.insert(tk.END, f"模型加载失败：{str(e)}\n\n")
            console_display.see(tk.END)

    def unload_model():
        global pipe
        try:
            pipe = None
            console_display.insert(tk.END, "模型已成功卸载！\n\n")
            console_display.see(tk.END)
            toggle_buttons(False)
            update_send_button()
        except Exception as e:
            console_display.insert(tk.END, f"模型卸载失败：{str(e)}\n\n")
            console_display.see(tk.END)

    def send_message():
        user_input = user_entry.get()
        if user_input.strip().lower() == 'quit':
            root.destroy()
            return

        chat_display.insert(tk.END, f"用户: {user_input}\n\n")
        chat_display.see(tk.END)
        user_entry.delete(0, tk.END)
        root.update()  # 强制更新界面，确保用户消息立即显示

        send_button.config(state=tk.DISABLED)  # 禁用发送按钮，防止重复点击
        root.update()  # 强制更新界面，确保按钮状态立即生效

        console_display.insert(tk.END, "消息成功发送，等待输出中......\n")
        console_display.see(tk.END)
        root.update()  # 强制更新界面，确保控制台消息立即显示

        try:
            prompt = f"User's question: '{user_input}'; AI assistant's answer: "
            result = pipe.generate([prompt], max_new_tokens=128, do_sample=False)
            perf_metrics = result.perf_metrics
            chat_display.insert(tk.END, f"助手: {result}\n\n")
            chat_display.see(tk.END)
            console_display.insert(tk.END, f"已成功输出，速度为 {perf_metrics.get_throughput().mean:.2f} tokens/s\n\n")
            console_display.see(tk.END)
        except Exception as e:
            chat_display.insert(tk.END, f"助手: 无法生成回复，错误: {str(e)}\n\n")
            chat_display.see(tk.END)
        finally:
            update_send_button()  # 恢复发送按钮状态

    def update_send_button(*args):
        is_enabled = user_entry.get().strip() and pipe
        send_button.config(state=tk.NORMAL if is_enabled else tk.DISABLED)
        user_entry.config(state=tk.NORMAL if pipe else tk.DISABLED)

    def on_closing():
        unload_model()
        root.destroy()

    root = tk.Tk()
    root.title("LLM 聊天助手")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 模型选择
    tk.Label(root, text="选择模型:").grid(row=0, column=0, padx=10, pady=10)
    model_var = tk.StringVar(value=model_list[0])
    model_menu = ttk.OptionMenu(root, model_var, *model_list)
    model_menu.grid(row=0, column=1, padx=10, pady=10)

    # 设备选择
    tk.Label(root, text="选择设备:").grid(row=1, column=0, padx=10, pady=10)
    device_var = tk.StringVar(value=device_list[0])
    device_menu = ttk.OptionMenu(root, device_var, device_list[0], *device_list)
    device_menu.config(width=12)  # 增加宽度以确保所有选项始终可见
    device_menu.grid(row=1, column=1, padx=10, pady=10)

    # 加载按钮
    load_button = ttk.Button(root, text="加载模型", command=load_model)
    load_button.grid(row=2, column=0, columnspan=2, pady=10)

    # 卸载按钮
    unload_button = ttk.Button(root, text="卸载模型", command=unload_model)
    unload_button.grid(row=3, column=0, columnspan=2, pady=10)
    unload_button.grid_remove()  # 初始隐藏卸载按钮

    # 聊天显示框
    chat_display = tk.Text(root, height=15, width=50, state=tk.NORMAL)
    chat_display.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    # 控制台显示框
    console_display = tk.Text(root, height=10, width=50, state=tk.NORMAL, bg="lightgray")
    console_display.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    # 用户输入框
    user_entry = ttk.Entry(root, width=40)
    user_entry.grid(row=6, column=0, padx=10, pady=10)
    user_entry.bind("<KeyRelease>", lambda event: update_send_button())
    user_entry.bind("<FocusIn>", lambda event: update_send_button())  # 确保每次输入框获得焦点时更新按钮状态

    # 发送按钮
    send_button = ttk.Button(root, text="发送", command=send_message, state=tk.DISABLED)
    send_button.grid(row=6, column=1, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
