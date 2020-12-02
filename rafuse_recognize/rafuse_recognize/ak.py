import tkinter as tk  # 使用Tkinter前需要先导入
import tkinter.filedialog

import tkinter
import os
# 第1步，实例化object，建立窗口window
window = tk.Tk()

# 第2步，给窗口的可视化起名字
window.title('My Window')

# 第3步，设定窗口的大小(长 * 宽)
window.geometry('500x300')  # 这里的乘是小x



# 定义一个函数功能（内容自己自由编写），供点击Button按键时调用，调用命令参数command=函数名
on_hit = False
var_load = tk.StringVar()

def wai():
	os.system('python ui.py')
	
def wai2():
	os.system('python ui2.py')

def shibie(load):
    os.system('python rafuse.py --image_file '+load)

def xunliang():
    os.system('python textcnn/train.py')

def pinggu():
    os.system('python textcnn/eval.py')

def ceshi(text):
    os.system('python textcnn/predict.py '+text)





def windows2():
    global on_hit
    if on_hit == False:
        on_hit =True
        window2 = tk.Tk()

        # 第2步，给窗口的可视化起名字
        window2.title('My Window')

        # 第3步，设定窗口的大小(长 * 宽)
        window2.geometry('500x300')
        submit_button = tk.Button(window2, text="模型训练", font=("微软雅黑", 20), command=xunliang).place(x=162, y=40)
        submit_button = tk.Button(window2, text="模型评估", font=("微软雅黑", 20), command=pinggu).place(x=162, y=120)
        submit_button = tk.Button(window2, text="单句测试", font=("微软雅黑", 20), command=wai2).place(x=162, y=200)
        window2.mainloop()



# 第5步，在窗口界面设置放置Button按键
submit_button = tk.Button(window, text="垃圾分类映射", font=("微软雅黑", 20), command=windows2).place(x=162, y=60)
submit_button = tk.Button(window, text="垃圾分类识别", font=("微软雅黑", 20), command=wai).place(x=162, y=160)

# 第6步，主窗口循环显示
window.mainloop()