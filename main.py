import  numpy as np
import  matplotlib.pyplot as plt
import tensorflow as tf
from django.contrib.admin import action
from django.utils.translation.template import plural_re
from tensorflow import  keras
from  tensorflow.keras import layers
from  sklearn.metrics import classification_report , confusion_matrix
import  seaborn as sns
import  os

from tensorflow.python.keras.combinations import keras_model_type_combinations

MODEL_PATH = "models/fashion_mnist_baseline.h5"
MODEL_PATH2 = "models/fashion_mnist_best.h5"
MODEL_PATH3 = "models/fashion_mnist_super.h5"
BASELINEMODE = "super"



print("TensorFlow version", tf.__version__)

# ==================== WEEK 1: DATA EXPLORATION ====================
print("\n=== Loading Fashion MNIST Dataset ===")
(x_train , y_train)  , (x_test , y_test) = keras.datasets.fashion_mnist.load_data()


class_names  = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']




print(f"Training samples  : {x_train.shape[0]}")
print(f"train shape     : {x_train.shape}")
print(f"test sampels : {x_test.shape[0]}")
print(f"Image shape {x_train.shape[1:]}")
print(f"Number of classes : {len(class_names)}")


def plot_samples(x, y , class_names , n_samples = 25):
    for i in range(n_samples):
        plt.figure(figsize=(10, 10))
        plt.subplot(5,5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i] , cmap='grey')
        plt.xlabel(class_names[y[i]])

    plt.tight_layout()
    plt.savefig('results/sample_images.png')
    print("\n✓ Sample images saved to 'results/sample_images.png'")
    plt.close()




# Data preprocessing
print("\n=== Preprocessing Data ===")
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train  = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print(f"Preprocessed training shape: {x_train.shape}")
print(f"Preprocessed test shape: {x_test.shape}")


# ==================== WEEK 2: BUILD BASELINE MODEL ====================
# calculations
"""
  Output Size = (Input Size - Kernel Size + 2×Padding) / Stride + 1
  Output Height = (28 - 3 + 2×0) / 1 + 1
       = (28 - 3) / 1 + 1
       = 25 / 1 + 1
       = 25 + 1
       = 26
"""


def create_baseline_model():
     model =  keras.Sequential([
     layers.Conv2D(32,(3,3) , activation = 'relu', input_shape = (28,28,1)),
     layers.MaxPooling2D((2,2)),
      # Second Convolutional Block

     layers.Conv2D(64 , (3,3)  , activation='relu'),
     layers.MaxPooling2D((2,2)),

     #third conventional layer
     layers.Conv2D(64 , (3,3) , activation='relu' ),
     layers.Flatten(),
     layers.Dense(64  , activation='relu'),
     layers.Dropout(0.5),
     #softmax because have more than two different lables to predict
     layers.Dense(10, activation='softmax' )



     ])



     #compiling the model
     model.compile(
     optimizer="adam",
     loss="sparse_categorical_crossentropy",
     metrics=['accuracy']
     )


     print("\n Model compiled\n")
     return  model


# Train improved model
def Trainand_run_baseline_model(baseline_model):


    print("=" * 50)
    print("\nModel Summary\n")
    baseline_model.summary()
    print("=" * 50)
    print("\n=== Training Baseline Model ===")

    history_baseline_model = baseline_model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=1,
    )
    # Evaluate baseline model
    print("\n=== Evaluating Baseline Model ===")
    test_loss, test_acc = baseline_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save baseline model
    baseline_model.save(MODEL_PATH)
    print(f"✓ Baseline model saved to {MODEL_PATH}\n")
    y_pred = baseline_model.predict(x_test)
    y_pred_classes  =  np.argmax(y_pred, axis=1)

    #plottings
    plot_training_history(history_baseline_model , "Baseline Model Progress"  , "baselinemodelprogress.png")
    plot_confusion_matrix(y_test , y_pred_classes , class_names)

def create_improved_model():
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    model  = keras.Sequential([


        data_augmentation,
        layers.Conv2D(32, (3,3) , activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),



        layers.Conv2D(64, (3,3) , activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3) , activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu' ),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax' )


    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss = 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    return  model

def call_backes(supr):


  callbacks = [
      keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=5,
          restore_best_weights=True

      ),


      keras.callbacks.ModelCheckpoint(
          MODEL_PATH3 if supr else MODEL_PATH2,
          monitor='val_accuracy',
          mode = 'max',
          save_best_only=True,
          verbose=1

      ),

      keras.callbacks.ReduceLROnPlateau(
          monitor='val_loss',
          factor=0.5,
          patience= 3 ,
          min_lr=  1e-7,
          verbose=1

      )
  ]

  return callbacks



# Train improved model
def Trainand_run_improved_model(callbacks , improved_model):
    print("\n=== Training Improved Model ===")
    history_improved  = improved_model.fit(
    x_train, y_train,
    epochs= 30,
    batch_size=128,
    validation_split  = 0.2,
    callbacks = callbacks,
    verbose=1

    )

    print("=" * 50)
    print("\nModel Summary\n")
    improved_model.summary()
    print("=" * 50)
    print("\n=== Training Baseline Model ===")


    # Evaluate baseline model
    print("\n=== Evaluating Improved  Model ===")
    test_loss, test_acc = improved_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save baseline model
    improved_model.save(MODEL_PATH)
    print(f"✓ Improved  model saved to {MODEL_PATH2}\n")

    y_pred  = improved_model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # plottings
    plot_training_history(history_improved, "Improved Model Progress", "improvedmodelprogress.png")
    plot_confusion_matrix(y_test, y_pred_classes, class_names)



def create_super_model():
    """
        Enhanced Super CNN Model for Fashion-MNIST
        - Residual connections
        - Deeper architecture
        - Regularization + BatchNorm
        """
    #enter the inputs as the 28,28 gray  scale image
    inputs   = keras.Input(shape=(28,28,1))


    #data rotation and the zooming
    x = layers.RandomRotation(0.1)(inputs)
    x = layers.RandomZoom(0.1)(x)

    #layer 1
    x = layers.Conv2D(32,(3,3)  , padding='same' ,  kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D((2,2))(x)


    #layer 2

    shortcut = x

    x =  layers.Conv2D(64 , (3,3)  , padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x  = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)


    shortcut  =  layers.Conv2D(64 , (1,1) , padding='same')(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)


    #layers 3

    x = layers.Conv2D(128 , (3,3) , padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x  = layers.MaxPooling2D((2,2))(x)

    #classifier

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256 , activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output= layers.Dense(10 , activation='softmax')(x)



    model  =  keras.models.Model(inputs ,  output)


    model.compile(
        optimizer=keras.optimizers.Adam(   learning_rate=1e-3),


        loss= "sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n✓ Super Model Compiled Successfully\n")

    return model


def train_super_model():
   model =  create_super_model()
   model.summary()

   history = model.fit(
       x_train, y_train,
       epochs= 40,
       batch_size=128,
       validation_split = 0.2,
       callbacks = call_backes(supr=True),
       verbose=1




   )

   test_loss , test_acc = model.evaluate(x_test , y_test , verbose=0)
   print(f"\n🚀 Super Model Test Accuracy: {test_acc:.4f}")
   model.save(MODEL_PATH3)
   print("✓ Super model saved")

   y_pred  =  model.predict(x_test)
   y_pred_class =  np.argmax(y_pred , axis=1)
   plot_training_history(history  , "Super Model Progress"  , "supermodelprogress.png")
   plot_confusion_matrix(y_test, y_pred_class, class_names)












def plot_training_history(history, title , filename ):
    fig , (ax1 , ax2)  = plt.subplots(1,2 ,figsize = (14,5))

    #accuracy
    ax1.plot(history.history['accuracy']  , label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{title} - Accuracy')
    ax1.legend()
    ax1.grid(True)

    #Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{title} - Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"results/{filename}" , dpi=150 , bbox_inches='tight')
    print(f"✓ Training history saved to 'results/{filename}'")
    plt.close()

def plot_confusion_matrix(y_true , y_pred , class_names):
    cm =  confusion_matrix(y_true,  y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm , annot= True ,  fmt= 'd'  , cmap= 'Blues' , xticklabels= class_names  ,  yticklabels= class_names )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png' , dpi=150 , bbox_inches='tight')
    print("\n=== Confusion Matrix Plot ===")
    plt.close()

# Evaluate improved model


if __name__ == "__main__":


    if BASELINEMODE == 'super':
        print("Creating super model")
        create_super_model()
        train_super_model()


    elif BASELINEMODE == 'true':
        create_baseline_model()
        Trainand_run_baseline_model(create_baseline_model())

    else:
        create_improved_model()
        Trainand_run_improved_model(call_backes(supr=False),create_improved_model())





