import suanpan
from flask import Flask, request, render_template
from suanpan import g
from suanpan.app import app
from suanpan.utils import json
from suanpan.storage import storage
from suanpan.app.arguments import String, Json


def saveParams(params):
    paramsFileKey = storage.getKeyInNodeConfigsStore("params.json")
    paramsFilePath = storage.getPathInNodeConfigsStore("params.json")
    json.dump(params, paramsFilePath)
    storage.upload(paramsFileKey, paramsFilePath)


def loadParams():
    paramsFilePath = storage.getPathInNodeConfigsStore("params.json")
    return json.load(paramsFilePath)


def create_app():
    # create and configure the app
    web = Flask(__name__)

    # a simple page that says hello
    @web.route('/')
    def hello():
        return render_template('pure.html')

    @web.route('/tmp_param', methods=['POST'])
    def tmp_param():
        # 存储临时变量到 g，可以和消息事件共享
        g.someParameter = 'a'

    @web.route('/storage_param', methods=['POST'])
    def storage_param():
        # 存储配置到oss，组件重启之后可以load
        params = request.get_json()
        saveParams(params)

    return web


def runFlask():
    web = create_app()
    app._stream.sioLoop.setWebApp(web)


@app.afterInit
def afterInit(context):
    try:
        # 从oss读取保存的参数配置
        params = loadParams()
    except:
        pass

    # 在sdk中运行flask，会自动分配端口
    runFlask()


@app.input(Json(key="inputData1", alias="user_text", default="Suanpan"))
@app.param(String(key="param_prefix", alias="prefix"))
@app.output(Json(key="outputData1", alias="result"))
def hello_world(context):
    args = context.args
    return f'Hello World, {args.prefix} {args.user_text}!'


if __name__ == "__main__":
    suanpan.run(app)
