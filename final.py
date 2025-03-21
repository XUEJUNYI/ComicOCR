import os
import time
import fitz
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
from PIL import Image
import os
from tqdm import tqdm
from yolo import YOLO
import shutil


OPENAI_API_ENDPOINT = None
OPENAI_API_KEY = None
API_VERSION = None
DEPLOYMENT_MODEL = None
VISION_ENDPOINT = None
VISION_KEY = None

with open("EndpointandKey.txt" ,"r") as f:
    lines  =f.readlines()
if len(lines) >= 6:
    OPENAI_API_ENDPOINT = lines[0].strip()  
    OPENAI_API_KEY = lines[1].strip()      
    API_VERSION = lines[2].strip()         
    DEPLOYMENT_MODEL = lines[3].strip()    
    VISION_ENDPOINT = lines[4].strip()     
    VISION_KEY = lines[5].strip()


# 全局变量
TOTAL_REQUESTS = 0
TOTAL_COST = 0.0
PRICE_PER_REQUEST_OCR = 0.001  

try:
    endpoint = VISION_ENDPOINT
    key = VISION_KEY
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

#建立客戶端
ocr_client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# OCR相關操作
def ocrpng(input_dir):
    resultlist = []
    global TOTAL_REQUESTS, TOTAL_COST
    def check_imgsize(image_path):
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 50 or height < 50 or width > 16000 or height > 16000:
                img = img.resize((50, 50), Image.Resampling.LANCZOS)
                img.save(image_path)
    try :
        for root, _, files in os.walk(input_dir):
            for file_name in tqdm(files):
                print(f"正在處理",root,file_name)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file_name)
                    check_imgsize(image_path)
                    with open(image_path, 'rb') as image:
                        image_rb = image.read()

                    results = ocr_client.analyze(
                        image_data=image_rb,
                        visual_features=["Read"]
                    )
                    
                    TOTAL_REQUESTS += 1
                    resultlist.append((results, image_path))

                    request_cost = PRICE_PER_REQUEST_OCR
                    TOTAL_COST += request_cost

        print(f"累計請求次数: {TOTAL_REQUESTS}")
        print(f"累計總花費: ${TOTAL_COST:.6f}")
    except Exception as e:
        error_message = f"Error processing file: {file_name} in {root}\nException: {str(e)}"
        print(error_message)
                    
        # 将错误记录到日志文件
        error_log_path = "ocr_error_log.txt"
        with open(error_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{error_message}\n")
            
    return resultlist

#轉換輸出格式
def turntxt(result, image_path):
    relative_path = os.path.relpath(image_path, "./input")
    
    txt_folder = os.path.join("./outputtxt", os.path.dirname(relative_path))
    os.makedirs(txt_folder, exist_ok=True)

    # 文件名和路径
    txt_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(txt_folder, f"{txt_name}.txt")

    # 提取 OCR 结果并保存到 txt 文件
    if result.read and result.read.blocks:
        text = []
        #text = [line['text'] for line in result.read.blocks[0].lines]
        for block in result.read.blocks:
            if hasattr(block ,'lines'):
                text.extend(line['text'] for line in block.lines)
        
        with open(txt_path, "w", encoding="utf-8") as w:
            for t in text:
                w.write(f"{t}\n")

    print(f"OCR 结果已保存到: {txt_path}")

#開始執行GPT相關操作
TOTAL_TOKEN = 0
PRICE_PER_TOKEN = 0.005 / 1000
REQUEST_COUNT = 0

client = AzureOpenAI(api_version=API_VERSION, 
                     azure_endpoint=OPENAI_API_ENDPOINT, 
                     api_key=OPENAI_API_KEY)
#讀取格式
def getnormal():
    with open("./normal.txt", "r", encoding="utf-8") as r:
        return "".join(r.readlines())

#合併文字檔(要求GPT回傳格式)
def get_merged_content(outputtxt_dir):
    merged_content = getnormal() + "\n\n" 
    for root, _, files in os.walk(outputtxt_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                merged_content += f.read() + "\n\n"
    return merged_content

#調用GPT操作
def request_test2(content, output_path):
    try:
        global TOTAL_TOKEN, PRICE_PER_TOKEN, REQUEST_COUNT
        completion = client.chat.completions.create(
            model=DEPLOYMENT_MODEL,
            temperature=0.7,
            max_tokens=1500,
            top_p=0.95,
            messages=[
                {"role": "system", "content": "You are an AI assistant"},
                {"role": "user", "content": content},
            ],
        )
        
        TOTAL_TOKEN += completion.usage.prompt_tokens * 3
        REQUEST_COUNT += 1
        json_data = completion.choices[0].message
        
        if json_data.content :
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as w:
                w.write(json_data.content)

        print(f"Price ($): {PRICE_PER_TOKEN * TOTAL_TOKEN}")
        print(f"Request Count: {REQUEST_COUNT}")
        
    except Exception as e:
        error_message = f"Error files：{output_path}\nError while processing content: {content[:50]}... (truncated)\nException: {str(e)}"
        print(error_message)
        
        # 将错误记录到日志文件
        error_log_path = "error_log.txt"
        with open(error_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{error_message}\n")
            
            
#調用切割PDF圖片方法
def pdf_to_png(file,pdf_path, output_folder,dpi=300):
    count = 0
    pdf_document = fitz.open(pdf_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_number in range(len(pdf_document)):

        page = pdf_document.load_page(page_number)
        zoom = dpi/72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
    
        output_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(output_path)
        
        
        count = count +1
        
    #計算預計花費
    total = count*15*0.001
    print(f"漫畫{file}預估需花費OCR費用：{round(total,3)}")
    total = total+0.075
    print(f"總計費用：{round(total,3)}美金\n")
    pdf_document.close()
    return total


def clear_folder():
    folders = [
        "img/",
        "img_out/",
        "./input",
        "./outputtxt",
        "./gptoutput",
        "./final"
    ]
    for folder_path in folders:
        if not os.path.exists(folder_path):
            print(f"資料夾不存在: {folder_path}")
            continue

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path) 
                print(f"删除文件: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"刪除空資料夾: {item_path}")



def request_test3(content, output_path):
    global TOTAL_TOKEN, PRICE_PER_TOKEN, REQUEST_COUNT
    completion = client.chat.completions.create(
        model=DEPLOYMENT_MODEL,
        temperature=0.7,
        max_tokens=1500,
        top_p=0.95,
        messages=[
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": content},
        ],
    )
    
    TOTAL_TOKEN += completion.usage.prompt_tokens * 3
    REQUEST_COUNT += 1
    json_data = completion.choices[0].message

    # 保存 GPT 结果
    if json_data.content :
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as w:
            w.write(json_data.content)
    else:
        print(json_data)
        print("Error: GPT response content is empty or missing.")
    print(f"Price ($): {PRICE_PER_TOKEN * TOTAL_TOKEN}")
    print(f"Request Count: {REQUEST_COUNT}")
   
def merge_txt_files(gptoutput_dir, finalout_dir):

    def natural_sort_key(s):
        return [int(t) if t.isdigit() else t for t in s.split('_')[-1].split('.')]

    for comic_name in os.listdir(gptoutput_dir):
        comic_path = os.path.join(gptoutput_dir, comic_name)
        
        final_comic_path = os.path.join(finalout_dir, comic_name)
        os.makedirs(final_comic_path, exist_ok=True)
        
        final_txt_path = os.path.join(final_comic_path, "final.txt")
        
        merged_content = ""

        for page_dir in sorted(os.listdir(comic_path), key=natural_sort_key):
            page_path = os.path.join(comic_path, page_dir)
            output_txt_path = os.path.join(page_path, "output_result.txt")
            
            if os.path.isdir(page_path) and os.path.exists(output_txt_path):
                with open(output_txt_path, "r", encoding="utf-8") as f:
                    page_content = f.read()
                    merged_content += page_content + "\n"

        with open(final_txt_path, "w", encoding="utf-8") as f:
            f.write(merged_content)

        print(f"{comic_name} 合併完成，结果保存在: {final_txt_path}")


# 主程序
if __name__ == "__main__":
    print("如果不需要更改，將圖片放置於in資料夾底下，後續設定輸入Enter即可")
    dir_origin_path = input("請輸入放置\"漫畫圖片\" 資料夾路徑 (默認: img/): ").strip() or "img/"
    dir_save_path = input("請輸入\"框選結果輸出\"資料夾路徑 (默認: img_out/): ").strip() or "img_out/"
    dir_intput_path= input("請輸入\"切割圖片路徑\"資料夾路徑 (默認: input/): ").strip() or "input/"
    input_dir = input("請輸入\"切割圖片路徑\"資料夾路徑 (默認: ./input 注意! 請與同上一致): ").strip() or "./input"
    outputtxt_dir = input("請輸入 OCR 输出資料夾路徑 (默認: ./outputtxt): ").strip() or "./outputtxt"
    gptoutput_dir = input("請輸入 GPT 輸出資料夾路徑 (默認: ./gptoutput): ").strip() or "./gptoutput"
    finalout_dir= input("請輸入 final 輸出資料夾路徑 (默認: ./final): ").strip() or "./final"
    
    clear_folder()
    
    #圖片裁切
    for root ,dirs , files in os.walk("./in"):
        total  = 0
        for file in files:
            pdf_path = f"{root}/{file}" 
            output_folder = f"img/{file}"  
            total = total + pdf_to_png(file,pdf_path, output_folder)
        print(f"全部漫畫總計費用：{round(total,3)}美金")

    yolo = YOLO()
    
    #yolo操作
    for root, dirs, files in os.walk(dir_origin_path):
        for img_name in tqdm(files):
            #搜取圖片
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                #設定輸出位置
                relative_path = os.path.relpath(root, dir_origin_path)
                img_name_no_ext = os.path.splitext(img_name)[0]
                #保存OCR位置
                save_folder = os.path.join(dir_save_path, relative_path,img_name_no_ext)
                #Crop輸出位置
                out_folder = os.path.join(dir_intput_path ,relative_path,img_name_no_ext)
                #輸入圖片路徑
                image_path = os.path.join(root, img_name)
                image = Image.open(image_path)
                #YOLO框選操作
                print(save_folder)
                r_image = yolo.detect_image(image,out_folder)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                #YOLO框選圖片位置
                save_path = os.path.join(save_folder, img_name.replace(".jpg", ".png"))

                r_image.save(save_path, quality=95, subsampling=0)
    
    time.sleep(1.5)  
    
    #OCR操作
    resultlist = ocrpng(input_dir)
    #寫入檔案
    for result, image_path in resultlist:
        turntxt(result, image_path)
    
    time.sleep(1.5)  
    
    # 調用GPT相關函數使用
    for main_dir in tqdm(os.listdir(outputtxt_dir), desc="Processing folders"):
        main_dir_path = os.path.join(outputtxt_dir, main_dir)
        # 確保是資料夾
        if os.path.isdir(main_dir_path):
            for subdir in tqdm(os.listdir(main_dir_path)):
                subdir_dir_path = os.path.join(main_dir_path, subdir)
                print(f"正在處理",main_dir_path,subdir_dir_path)
                merged_content = get_merged_content(subdir_dir_path)
                #添加gptoutput輸出路徑
                gptsave_subdir = os.path.join(gptoutput_dir, main_dir , subdir)
                #創建該目錄
                os.makedirs(gptsave_subdir, exist_ok=True)
                #拼接出書目錄名稱
                output_file_path = os.path.join(gptsave_subdir, "output_result.txt")
                #調用GPT參數
                request_test2(merged_content, output_file_path)

    
    merge_txt_files(gptoutput_dir, finalout_dir)
    
    