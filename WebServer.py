from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        button_pressed = request.form.get('button_pressed', False)
        return render_template('result.html', button_pressed=button_pressed)

if __name__ == "__main__":
    app.run()
    #app.route