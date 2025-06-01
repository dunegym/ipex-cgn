from transformers import AutoTokenizer
import openvino_genai as ov_genai
import time
import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading

MODEL_LIST = ['TinyLlama-1.1B', 'DeepSeek-1.5B']
QUANTIZATION_LIST = ['int4', 'int8']
DEVICE_LIST = ['CPU', 'GPU', 'NPU']


class LLMChatManager:
    def clear_history(self):
        self.chat_history = []

    def __init__(self):
        self.pipe = None
        self.tokenizer = None
        self.chat_history = []

    def load_model(self, model_name, quant, device, console_callback=None):
        model_dir = f"model/{model_name}-{quant}"
        if console_callback:
            console_callback('Loading......\n')
        try:
            start_time = time.time()
            if device == 'NPU':
                if not os.path.exists('.npucache'):
                    os.makedirs('.npucache')
                self.pipe = ov_genai.LLMPipeline(
                    model_dir,
                    device,
                    CACHE_DIR=f".npucache/{model_name}-{quant}",
                    MAX_PROMPT_LEN=2048
                )
            else:
                self.pipe = ov_genai.LLMPipeline(model_dir, device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            end_time = time.time()
            load_time = end_time - start_time
            if console_callback:
                console_callback(f"模型加载成功！耗时：{load_time:.2f} 秒\n\n")
            return True
        except Exception as e:
            if console_callback:
                console_callback(f"模型加载失败：{str(e)}\n\n")
            self.pipe = None
            self.tokenizer = None
            return False

    def unload_model(self, console_callback=None):
        try:
            self.pipe = None
            self.tokenizer = None
            if console_callback:
                console_callback("模型已成功卸载！\n\n")
            return True
        except Exception as e:
            if console_callback:
                console_callback(f"模型卸载失败：{str(e)}\n\n")
            return False

    def build_prompt(self, user_input, model_name):
        model_dir = None
        for quant in QUANTIZATION_LIST:
            candidate = f"model/{model_name}-{quant}"
            if os.path.isdir(candidate):
                model_dir = candidate
                break
        if model_dir is None:
            model_dir = f"model/{model_name}-int4"  # 兜底

        try:
            n = len(self.chat_history)
            history = self.chat_history[-n:] if n <= 6 else self.chat_history[-6:]
            messages = [
                {"role": "system", "content": "You are a helpful assistant. "}
            ] + history + [{"role": "user", "content": user_input}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            return f"<|user|>\n{user_input}\n<|assistant|>\n"

    def generate_reply(self, prompt, max_new_tokens=512):
        if not self.pipe:
            raise RuntimeError("模型未加载")
        return self.pipe.generate(
            [prompt],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            use_cache=True
        )

    def append_history(self, user_input, assistant_output):
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": str(assistant_output)})
        if len(self.chat_history) > 6:
            self.chat_history = self.chat_history[-6:]

# GUI 主程序
def start_gui():
    def do_clear_history():
        manager.clear_history()
        console_callback("上下文已清空\n\n")
        update_send_button()

    manager = LLMChatManager()

    def toggle_buttons(is_loaded):
        if is_loaded:
            load_button.grid_remove()
            unload_button.grid()
            model_menu.config(state=tk.DISABLED)
            quant_menu.config(state=tk.DISABLED)
            device_menu.config(state=tk.DISABLED)
        else:
            unload_button.grid_remove()
            load_button.grid()
            model_menu.config(state=tk.NORMAL)
            quant_menu.config(state=tk.NORMAL)
            device_menu.config(state=tk.NORMAL)

    def console_callback(msg):
        console_display.insert(tk.END, msg)
        console_display.see(tk.END)
        root.update()

    def chat_callback(msg):
        chat_display.insert(tk.END, msg)
        chat_display.see(tk.END)
        root.update()

    def do_load_model():
        def load():
            selected_model = model_var.get()
            selected_quant = quant_var.get()
            selected_device = device_var.get()
            # 加载前先禁用按钮，防止重复点击
            root.after(0, lambda: toggle_buttons(False))
            root.after(0, lambda: update_send_button())
            # 分阶段输出，防止界面假死
            def safe_console(msg):
                root.after(0, lambda: console_callback(msg))
            success = manager.load_model(selected_model, selected_quant, selected_device, safe_console)
            root.after(0, lambda: toggle_buttons(success))
            root.after(0, update_send_button)
        threading.Thread(target=load, daemon=True).start()

    def do_unload_model():
        manager.unload_model(console_callback)
        toggle_buttons(False)
        update_send_button()

    def do_send_message():
        def send():
            user_input = user_entry.get()
            if user_input.strip().lower() == 'quit':
                root.destroy()
                return
            chat_callback(f"用户: {user_input}\n\n")
            user_entry.delete(0, tk.END)
            send_button.config(state=tk.DISABLED)
            console_callback("消息成功发送，等待输出中......\n")
            try:
                selected_model = model_var.get()
                prompt = manager.build_prompt(user_input, selected_model)
                result = manager.generate_reply(prompt)
                perf_metrics = result.perf_metrics
                chat_callback(f"助手: {result}\n\n")
                console_callback(f"已成功输出，速度为 {perf_metrics.get_throughput().mean:.2f} tokens/s\n\n")
                manager.append_history(user_input, result)
            except Exception as e:
                chat_callback(f"助手: 无法生成回复，错误: {str(e)}\n\n")
            finally:
                update_send_button()
        threading.Thread(target=send, daemon=True).start()

    def update_send_button(*args):
        is_enabled = user_entry.get().strip() and manager.pipe
        send_button.config(state=tk.NORMAL if is_enabled else tk.DISABLED)
        user_entry.config(state=tk.NORMAL if manager.pipe else tk.DISABLED)
        clear_button.config(state=tk.NORMAL if manager.pipe else tk.DISABLED)

    def on_closing():
        manager.unload_model()
        root.destroy()

    root = tk.Tk()
    root.title("LLM 聊天助手")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    tk.Label(root, text="选择模型:").grid(row=0, column=0, padx=10, pady=10)
    model_var = tk.StringVar(value=MODEL_LIST[0])
    def update_model_menu():
        menu = model_menu['menu']
        menu.delete(0, 'end')
        for m in MODEL_LIST:
            menu.add_command(label=m, command=tk._setit(model_var, m))
        model_var.set(MODEL_LIST[0])

    model_menu = ttk.OptionMenu(root, model_var, MODEL_LIST[0], *MODEL_LIST)
    model_menu.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(root, text="量化精度:").grid(row=1, column=0, padx=10, pady=10)
    quant_var = tk.StringVar(value=QUANTIZATION_LIST[0])
    quant_menu = ttk.OptionMenu(root, quant_var, QUANTIZATION_LIST[0], *QUANTIZATION_LIST)
    quant_menu.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="选择设备:").grid(row=2, column=0, padx=10, pady=10)
    device_var = tk.StringVar(value=DEVICE_LIST[0])
    device_menu = ttk.OptionMenu(root, device_var, DEVICE_LIST[0], *DEVICE_LIST)
    device_menu.config(width=12)
    device_menu.grid(row=2, column=1, padx=10, pady=10)


    load_button = ttk.Button(root, text="加载模型", command=do_load_model)
    load_button.grid(row=3, column=0, columnspan=2, pady=10)

    unload_button = ttk.Button(root, text="卸载模型", command=do_unload_model)
    unload_button.grid(row=4, column=0, columnspan=2, pady=10)
    unload_button.grid_remove()

    clear_button = ttk.Button(root, text="清空上下文", command=do_clear_history, state=tk.DISABLED)
    clear_button.grid(row=5, column=0, columnspan=2, pady=5)



    # 增大宽度 10%
    chat_display = tk.Text(root, height=15, width=55, state=tk.NORMAL)
    chat_display.grid(row=6, column=0, columnspan=2, padx=12, pady=10)

    console_display = tk.Text(root, height=10, width=55, state=tk.NORMAL, bg="lightgray")
    console_display.grid(row=7, column=0, columnspan=2, padx=12, pady=10)

    user_entry = ttk.Entry(root, width=44)
    user_entry.grid(row=8, column=0, padx=12, pady=10)
    user_entry.bind("<KeyRelease>", lambda event: update_send_button())
    user_entry.bind("<FocusIn>", lambda event: update_send_button())

    send_button = ttk.Button(root, text="发送", command=do_send_message, state=tk.DISABLED)
    send_button.grid(row=8, column=1, padx=12, pady=10)

    update_model_menu()

    root.mainloop()

if __name__ == "__main__":
    start_gui()
