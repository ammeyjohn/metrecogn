import sys
import os
sys.path.append(os.path.abspath('.'))

from werkzeug.utils import secure_filename
from flask import Flask, render_template, render_template_string, url_for, jsonify, request, request, jsonify, send_from_directory, abort, send_file
import time
import recognize as rcg

app = Flask(__name__, template_folder='template')
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['txt', 'png', 'jpg', 'xls', 'JPG', 'PNG', 'xlsx', 'gif', 'GIF'])


# 用于判断文件后缀
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/index')
def index():
    return render_template('index.html')


# 用于测试上传，稍后用到
@app.route('/test/upload')
def upload_test():
    return render_template('upload.html')


# 上传文件
@app.route('/api/upload', methods=['POST'], strict_slashes=False)
def api_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['myfile']  # 从表单的file字段获取文件，myfile为该表单的name值
    if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
        fname = secure_filename(f.filename)
        ext = fname.rsplit('.', 1)[1]  # 获取文件后缀
        unix_time = int(time.time())
        new_filename = str(unix_time) + '.' + ext  # 修改了上传的文件名
        f.save(os.path.join(file_dir, new_filename))  # 保存文件到upload目录

        result = {
            'name': new_filename,
            'url': 'static/upload/' + new_filename,
            'number': None,
            'info': None
        }
        print(result)

        # 解析读数
        numbers, info = rcg.recognize(os.path.join(file_dir, new_filename))
        if numbers is not None and len(numbers) > 0:
            result['number'] = numbers[0]
            result['info'] = info
        else:
            result['number'] = 'Cannot recognize'
        print(result)
        return jsonify({'files': result})

    else:
        return jsonify({"errno": 1001, "errmsg": "上传失败"})


@app.route('/api/download/<filename>')
def download(filename):
    if request.method == "GET":
        if os.path.exists(os.path.join(basedir, 'upload', filename)):
            return send_from_directory('upload', filename, as_attachment=False)
        abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
