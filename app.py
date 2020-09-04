{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled1.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOZI8vV/DanzK50GLUGR94v",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/alaa-samy/Celsuis-Fahrenheit/blob/master/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xkL1Slliu6OK",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#Import dependencies\n",
        "\n",
        "import tensorflow as tf\n",
        "import numpy as np \n",
        "import logging\n",
        "logger = tf.get_logger()\n",
        "logger.setLevel(logging.ERROR)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2WjpbjzbygYM",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 136
        },
        "outputId": "5d91f776-32a7-43eb-d338-b28258a25e3d"
      },
      "source": [
        "#Set up training data\n",
        "\n",
        "celsuis = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)\n",
        "fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)\n",
        "\n",
        "for i,c in enumerate(celsuis):\n",
        "  print(\"{} degrees Celsuis = {} degrees Fahrenheit\" . format(c, fahrenheit[i]))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "-40.0 degrees Celsuis = -40.0 degrees Fahrenheit\n",
            "-10.0 degrees Celsuis = 14.0 degrees Fahrenheit\n",
            "0.0 degrees Celsuis = 32.0 degrees Fahrenheit\n",
            "8.0 degrees Celsuis = 46.0 degrees Fahrenheit\n",
            "15.0 degrees Celsuis = 59.0 degrees Fahrenheit\n",
            "22.0 degrees Celsuis = 72.0 degrees Fahrenheit\n",
            "38.0 degrees Celsuis = 100.0 degrees Fahrenheit\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dt3jtfW813pV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#Create a model\n",
        "\n",
        "l0 = tf.keras.layers.Dense(units= 1, input_shape=[1])\n",
        "model = tf.keras.Sequential([l0])\n",
        "model.compile(loss = 'mean_squared_error' , optimizer = tf.keras.optimizers.Adam(0.1))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8oBLV0JV59Jl",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "a84152c2-2903-4143-883a-4e79ddb60f7d"
      },
      "source": [
        "#Train the model\n",
        "\n",
        "history = model.fit(celsuis , fahrenheit ,epochs = 500 , verbose = False)\n",
        "print('Finish the training')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Finish the training\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c0LN-yh166Qr",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "074e98e9-f836-4a5d-f01d-3754d7a978cb"
      },
      "source": [
        "print(model.predict([100.0]))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[211.7398]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rQdd6m1o7Z2Q",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "a0841d87-da96-4e32-e44b-35e46e3cab3a"
      },
      "source": [
        "print(\"That's our layer variable {}\" . format(l0.get_weights()))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "That's our layer variable [array([[1.7984569]], dtype=float32), array([31.89412], dtype=float32)]\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}