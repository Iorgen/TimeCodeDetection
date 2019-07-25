from flask import render_template, session, redirect, url_for, current_app, request
from core.controller import TimeCodeController
from .forms import RequestForm
from . import main


@main.route('/', methods=['POST', 'GET'])
def index():
    name = None
    form = RequestForm()
    if form.validate_on_submit():
        name = form.name.data
        video = form.video.data
        predictions = TimeCodeController().video_recognition(video)
        print(predictions)
        return render_template("results.html", predictions=predictions,  init=True)
    return render_template('index.html', form=form)


# Telefram bot will working throw web hook, as example heroku
