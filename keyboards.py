from aiogram import types


def start_keyboard():
    buttons = [
        types.InlineKeyboardButton(text="Отправить фото", callback_data="send_photo")
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*buttons)

    return keyboard


def select_style():
    buttons = [
        types.InlineKeyboardButton(text="Негатив", callback_data="filter_1"),
        types.InlineKeyboardButton(text="Серое", callback_data="filter_2"),
        types.InlineKeyboardButton(text="Монохром", callback_data="filter_3"),
        types.InlineKeyboardButton(text="Контурное", callback_data="filter_4"),
        types.InlineKeyboardButton(text="Фильтр Габора", callback_data="filter_6"),
        types.InlineKeyboardButton(text="Отправь новое фото", callback_data="send_photo")
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(*buttons)

    return keyboard


start_message = ("Привет! \U0001F44B\n\n"
                 "Я умею обрабатывать изображения и накладывать на них фильтры\n\n"
                 "Если станет интересно, как это все работает, "
                 "то можешь ознакомиться со страницей проекта на GitHub")

menu_message = ("Ознакомься с каждым из фильтров!\n\n"
                "Если я неправильно работаю, то попробуй перезапустить меня командой /start")