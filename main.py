from transformers import AutoTokenizer
import openvino_genai as ov_genai
import time
import tkinter as tk
from tkinter import ttk, messagebox
import os


model_list=['TinyLlama-1.1B',
            'DeepSeek-1.5B']
quantization_list=['int4','int8']
device_list=['CPU','GPU','NPU']
pipe = None  # 确保在程序启动时 pipe 被定义
chat_history = []

# GUI 主程序
def start_gui():
    def toggle_buttons(is_loaded):
        if is_loaded:
            load_button.grid_remove()
            unload_button.grid()
            model_menu.config(state=tk.DISABLED)  # 禁用模型选项卡
            quant_menu.config(state=tk.DISABLED)  # 禁用量化精度选项卡
            device_menu.config(state=tk.DISABLED)  # 禁用设备选项卡
        else:
            unload_button.grid_remove()
            load_button.grid()
            model_menu.config(state=tk.NORMAL)  # 启用模型选项卡
            quant_menu.config(state=tk.NORMAL)  # 启用量化精度选项卡
            device_menu.config(state=tk.NORMAL)  # 启用设备选项卡

    def load_model():
        selected_model = model_var.get()
        selected_quant = quant_var.get()
        selected_device = device_var.get()
        model_dir = f"model/{selected_model}-{selected_quant}"
        console_display.insert(tk.END, 'Loading......\n')
        console_display.see(tk.END)  # 确保最新消息可见
        root.update()  # 强制更新界面，确保立即显示 'Loading......'
        try:
            start_time = time.time()
            if selected_device == 'NPU':
                # 确保 .npucache 文件夹存在
                if not os.path.exists('.npucache'):
                    os.makedirs('.npucache')
                global pipe
                pipe = ov_genai.LLMPipeline(model_dir, 
                                            selected_device, 
                                            GENERATE_HINT="BEST_PERF", 
                                            CACHE_DIR=f".npucache/{selected_model}-{selected_quant}",
                                            MAX_PROMPT_LEN=2048
                )
            else:
                pipe = ov_genai.LLMPipeline(model_dir, selected_device)

            # 加载tokenizer，供后续build_prompt使用
            global tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
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

    def build_prompt(user_input, model_name):
        """
        根据模型名自动选择模板，优先使用 transformers tokenizer 的 apply_chat_template 方法。
        支持上下文记忆，自动拼接最近3轮对话。
        """
        model_dir = None
        for quant in quantization_list:
            candidate = f"model/{model_name}-{quant}"
            if os.path.isdir(candidate):
                model_dir = candidate
                break
        if model_dir is None:
            model_dir = f"model/{model_name}-int4"  # 兜底

        try:
            global tokenizer, chat_history
            # 取最近3轮历史，每轮包含user和assistant
            n = len(chat_history)
            history = chat_history[-n:] if n <= 6 else chat_history[-6:]
            # 构造messages，历史+当前user
            messages = [{"role": "system", "content": "You are a helpful assistant. "}] + history + [{"role": "user", "content": user_input}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 回退到手动模板
            return f"<|user|>\n{user_input}\n<|assistant|>\n"


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
            selected_model = model_var.get()
            prompt = build_prompt(user_input, selected_model)
            result = pipe.generate(
                [prompt],
                max_new_tokens=512,
                do_sample=True,
                use_cache=True
            )
            perf_metrics = result.perf_metrics
            chat_display.insert(tk.END, f"助手: {result}\n\n")
            chat_display.see(tk.END)
            console_display.insert(tk.END, f"已成功输出，速度为 {perf_metrics.get_throughput().mean:.2f} tokens/s\n\n")
            console_display.see(tk.END)
            # 记录历史对话，最多保留最近3轮
            global chat_history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": str(result)})
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]
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
    def update_model_menu():
        menu = model_menu['menu']
        menu.delete(0, 'end')
        for m in model_list:
            menu.add_command(label=m, command=tk._setit(model_var, m))
        model_var.set(model_list[0])

    model_menu = ttk.OptionMenu(root, model_var, model_list[0], *model_list)
    model_menu.grid(row=0, column=1, padx=10, pady=10)

    # 量化精度选择
    tk.Label(root, text="量化精度:").grid(row=1, column=0, padx=10, pady=10)
    quant_var = tk.StringVar(value=quantization_list[0])
    quant_menu = ttk.OptionMenu(root, quant_var, quantization_list[0], *quantization_list)
    quant_menu.grid(row=1, column=1, padx=10, pady=10)

    # 设备选择
    tk.Label(root, text="选择设备:").grid(row=2, column=0, padx=10, pady=10)
    device_var = tk.StringVar(value=device_list[0])
    device_menu = ttk.OptionMenu(root, device_var, device_list[0], *device_list)
    device_menu.config(width=12)  # 增加宽度以确保所有选项始终可见
    device_menu.grid(row=2, column=1, padx=10, pady=10)


    # 加载按钮
    load_button = ttk.Button(root, text="加载模型", command=load_model)
    load_button.grid(row=3, column=0, columnspan=2, pady=10)

    # 卸载按钮
    unload_button = ttk.Button(root, text="卸载模型", command=unload_model)
    unload_button.grid(row=4, column=0, columnspan=2, pady=10)
    unload_button.grid_remove()  # 初始隐藏卸载按钮


    # 聊天显示框
    chat_display = tk.Text(root, height=15, width=50, state=tk.NORMAL)
    chat_display.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    # 控制台显示框
    console_display = tk.Text(root, height=10, width=50, state=tk.NORMAL, bg="lightgray")
    console_display.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    # 用户输入框
    user_entry = ttk.Entry(root, width=40)
    user_entry.grid(row=7, column=0, padx=10, pady=10)
    user_entry.bind("<KeyRelease>", lambda event: update_send_button())
    user_entry.bind("<FocusIn>", lambda event: update_send_button())  # 确保每次输入框获得焦点时更新按钮状态

    # 发送按钮
    send_button = ttk.Button(root, text="发送", command=send_message, state=tk.DISABLED)
    send_button.grid(row=7, column=1, padx=10, pady=10)


    # 保证模型选项卡内容与 model_list 同步
    update_model_menu()

    root.mainloop()

if __name__ == "__main__":
    start_gui()
