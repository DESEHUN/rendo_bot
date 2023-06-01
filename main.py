import logging
from aiogram import Bot, Dispatcher, executor, types
import keyboards as kb
import warnings
import imageutils
import settings
from dotenv import load_dotenv
import os
warnings.filterwarnings("ignore")

load_dotenv()
BOT_TOKEN = os.getenv('TOKEN')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
db_photos = {}


@dp.message_handler(commands=['start', 'help'])  # стартовое сообщение
async def send_welcome(message: types.Message):
    db_photos[message.from_user.id] = 0
    logging.info(f"New User! Current number of users in dict: {len(db_photos)}")
    await message.answer(kb.start_message, reply_markup=kb.start_keyboard())


@dp.callback_query_handler(text="send_photo")  # запрос фото
async def transfer_style(call: types.CallbackQuery):
    await call.message.answer("Мне нужна 1 фотография. Отправь мне ее.\n")
    await call.answer()


@dp.message_handler(content_types=["photo"])  # сохранение фото в словарь
async def get_photo(message):
    file_id = message.photo[-1].file_id
    db_photos[message.from_user.id] = file_id
    logging.info(f"Users right now: {db_photos}")
    await message.photo[-1].download(destination_file=f'C:/Users/kuzma/Desktop/photo_bot/{message.from_user.id}.jpg')
    await message.answer(kb.menu_message, reply_markup=kb.select_style())


@dp.callback_query_handler(text="filter_1")
async def filter_1(call):
    img = imageutils.ImageProcessor(f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')
    res_path = os.path.splitext(os.path.join(settings.OUT_IMAGE_PATH, os.path.basename(
        f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')))[0] + '.png'
    img.negate()
    img.save(res_path)
    await bot.send_photo(call.message.chat.id, photo=open(res_path, 'rb'))
    await call.message.answer(kb.menu_message, reply_markup=kb.select_style())
    await call.answer()


@dp.callback_query_handler(text="filter_2")
async def filter_2(call):
    img = imageutils.ImageProcessor(f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')
    res_path = os.path.splitext(os.path.join(settings.OUT_IMAGE_PATH, os.path.basename(
        f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')))[0] + '.png'
    img.to_grayscale()
    img.save(res_path)
    await bot.send_photo(call.message.chat.id, photo=open(res_path, 'rb'))
    await call.message.answer(kb.menu_message, reply_markup=kb.select_style())
    await call.answer()


@dp.callback_query_handler(text="filter_3")
async def filter_3(call):
    img = imageutils.ImageProcessor(f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')
    res_path = os.path.splitext(os.path.join(settings.OUT_IMAGE_PATH, os.path.basename(
        f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')))[0] + '.png'
    img.to_binary()
    img.save(res_path)
    await bot.send_photo(call.message.chat.id, photo=open(res_path, 'rb'))
    await call.message.answer(kb.menu_message, reply_markup=kb.select_style())
    await call.answer()


@dp.callback_query_handler(text="filter_4")
async def filter_4(call):
    img = imageutils.ImageProcessor(f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')
    res_path = os.path.splitext(os.path.join(settings.OUT_IMAGE_PATH, os.path.basename(
        f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')))[0] + '.png'
    img.scharr_operator(threshold=75)
    img.save(res_path)
    await bot.send_photo(call.message.chat.id, photo=open(res_path, 'rb'))
    await call.message.answer(kb.menu_message, reply_markup=kb.select_style())
    await call.answer()


@dp.callback_query_handler(text="filter_6")
async def filter_6(call):
    img = imageutils.ImageProcessor(f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')
    res_path = os.path.splitext(os.path.join(settings.OUT_IMAGE_PATH, os.path.basename(
        f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg')))[0] + '.png'
    imageutils.gabor_filter(f'C:/Users/kuzma/Desktop/photo_bot/{call.message.chat.id}.jpg', res_path)
    img = open(res_path, 'rb')
    await bot.send_photo(call.message.chat.id, photo=open(res_path, 'rb'))
    await call.message.answer(kb.menu_message, reply_markup=kb.select_style())
    await call.answer()


async def on_shutdown(dp):
    logging.warning("Shutting down..")
    await dp.storage.close()
    await dp.storage.wait_closed()
    logging.warning("Bye!")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)