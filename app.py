from flask import Flask, render_template, request, redirect, send_from_directory
import json
import os
import threading


app = Flask(__name__)

app.config["TEMPLATES_AUTO_RELOAD"] = True


# =========================
# 路径
# =========================

DATA_JSONL = (
    "/home/wx1522744/zhaoqianqian/project/"
    "Complex_bench/candidates_balanced/"
    "candidate_5000_balanced.jsonl"
)


IMAGE_DIR = (
    "/home/wx1522744/zhaoqianqian/project/"
    "Complex_bench/candidates_balanced/images"
)


OUTPUT_DIR = (
    "/home/wx1522744/zhaoqianqian/project/"
    "Complex_bench/annotator/output"
)


ANNOTATION_DIR = os.path.join(
    OUTPUT_DIR,
    "annotations"
)


STATUS_PATH = os.path.join(
    OUTPUT_DIR,
    "task_status.json"
)


DRAFT_PATH = os.path.join(
    OUTPUT_DIR,
    "draft_status.json"
)


os.makedirs(
    ANNOTATION_DIR,
    exist_ok=True
)



# =========================
# 分类体系
# =========================

CATEGORY_MAP={

"文档类":[
"邀请函",
"请帖",
"节日祝福",
"致谢信",
"简历",
"工作报告",
"其他"
],

"幻灯片类":[
"PPT",
"课件",
"模板",
"汇报材料",
"其他"
],

"海报类":[
"招聘海报",
"电影海报",
"公益海报",
"商业广告",
"活动海报",
"美食海报",
"其他"
],

"环境文字类":[
"地铁",
"道路",
"门店",
"建筑",
"其他"
],

"界面类":[
"淘宝",
"小红书",
"APP",
"软件界面",
"网页",
"其他"
],

"信息图类":[
"医学",
"工程",
"科技",
"AI",
"数据图",
"其他"
],

"艺术文字类":[
"国画",
"油画",
"漫画",
"二次元",
"插画",
"其他"
],

"真实场景类":[
"会议",
"发布会",
"生活照片",
"风景",
"其他"
]

}



# =========================
# 加载数据
# =========================

samples=[]

with open(
    DATA_JSONL,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        samples.append(
            json.loads(line)
        )


print(
    "加载图片:",
    len(samples)
)




# =========================
# 状态
# =========================

lock=threading.Lock()



if os.path.exists(STATUS_PATH):

    with open(
        STATUS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        task_status=json.load(f)

else:

    task_status={}



if os.path.exists(DRAFT_PATH):

    with open(
        DRAFT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        drafts=json.load(f)

else:

    drafts={}



# 当前任务

user_task={}



def save_status():

    with open(
        STATUS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            task_status,
            f,
            ensure_ascii=False,
            indent=2
        )



def save_draft():

    with open(
        DRAFT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            drafts,
            f,
            ensure_ascii=False,
            indent=2
        )



# =========================
# 用户统计
# =========================

def get_user_count(user):

    path=os.path.join(
        ANNOTATION_DIR,
        user+".jsonl"
    )


    if not os.path.exists(path):

        return 0


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return sum(
            1 for _ in f
        )



# =========================
# 查找用户任务
# =========================

def get_user_task(user):


    for img_id,info in task_status.items():

        if (
            info["user"]==user
            and
            info["status"]=="doing"
        ):

            return info["item"]


    return None




# =========================
# 分配任务
# =========================

def get_task(user):


    current=get_user_task(user)

    if current:

        return current



    with lock:


        for item in samples:


            img_id=item["id"]


            if img_id not in task_status:


                task_status[img_id]={

                    "status":"doing",

                    "user":user,

                    "item":item

                }


                save_status()


                return item



    return None



# =========================
# 首页
# =========================

@app.route("/")
def index():


    user=request.args.get(
        "user",
        "user1"
    )


    item=get_task(user)


    if item is None:

        return "全部完成"



    return render_template(
        "index.html",
        item=item,
        categories=CATEGORY_MAP,
        user=user,
        user_count=get_user_count(user),
        total_count=len(samples),
        draft=drafts.get(user,{})
    )



# =========================
# 图片
# =========================

@app.route("/image/<filename>")
def image(filename):

    return send_from_directory(
        IMAGE_DIR,
        filename
    )



# =========================
# 检查重复
# =========================

def duplicated(user,img_id):


    path=os.path.join(
        ANNOTATION_DIR,
        user+".jsonl"
    )


    if not os.path.exists(path):

        return False


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:


        for line in f:

            if json.loads(line)["id"]==img_id:

                return True


    return False



# =========================
# 保存正式标注
# =========================

@app.route(
"/save",
methods=["POST"]
)
def save():


    user=request.form["user"]


    item=get_user_task(user)


    if item is None:

        return redirect(
            "/?user="+user
        )


    if duplicated(user,item["id"]):

        return redirect(
            "/?user="+user
        )


    result={

        "id":item["id"],

        "image":item["image"],

        "text":item["text"],

        "domain":request.form.get("domain"),

        "category":request.form.get("category"),

        "category_custom":request.form.get(
            "category_custom",
            ""
        ),

        "attributes":{

            "text_length":request.form.get("text_length"),

            "text_number":request.form.get("text_number"),

            "layout":request.form.get("layout"),

            "semantic":request.form.get("semantic")

        },

        "annotator":user

    }



    path=os.path.join(
        ANNOTATION_DIR,
        user+".jsonl"
    )


    with open(
        path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            )
            +"\n"
        )



    with lock:

        task_status[item["id"]]={
            "status":"done",
            "user":user,
            "item":item
        }

        save_status()



    if user in drafts:

        del drafts[user]

        save_draft()



    return redirect(
        "/?user="+user
    )


# =========================
# 保存草稿 + 上一张
@app.route(
"/previous",
methods=["POST"]
)
def previous():


    user=request.form["user"]


    item=get_user_task(user)


    if item is None:

        return redirect(
            "/?user="+user
        )


    drafts[user]={


        "id":
        item["id"],



        "domain":
        request.form.get(
            "domain"
        ),



        "category":
        request.form.get(
            "category"
        ),



        "category_custom":
        request.form.get(
            "category_custom",
            ""
        ),



        "text_length":
        request.form.get(
            "text_length"
        ),



        "text_number":
        request.form.get(
            "text_number"
        ),



        "layout":
        request.form.get(
            "layout"
        ),



        "semantic":
        request.form.get(
            "semantic"
        )


    }



    save_draft()



    return redirect(
        "/?user="+user
    )




# =========================
# 跳过
# =========================
@app.route("/skip")
def skip():


    user=request.args.get(
        "user",
        "user1"
    )


    item=get_user_task(user)



    if item:


        with lock:


            if item["id"] in task_status:

                del task_status[item["id"]]


            save_status()



    # 删除草稿

    if user in drafts:

        del drafts[user]

        save_draft()



    return redirect(
        "/?user="+user
    )



if __name__=="__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )

