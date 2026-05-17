import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from multi_user_adapter import MultiUserVoiceAdapter


def select_user():
    print("\n" + "=" * 50)
    print("AI SNAKE - ВЫБОР ПОЛЬЗОВАТЕЛЯ")
    print("=" * 50)

    adapter = MultiUserVoiceAdapter(save_dir="users/")
    users = adapter.get_user_list()

    if users:
        print("\nСуществующие пользователи:")
        for i, user in enumerate(users, 1):
            stats = adapter.get_stats().get(user, {})
            samples = stats.get('samples', 0)
            print(f"  {i}. {user} ({samples} образцов)")
        print(f"  {len(users) + 1}. Создать нового пользователя")

        choice = input("\nВыберите номер: ").strip()

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(users):
                user_id = users[choice_num - 1]
                print(f"\nВыбран пользователь: {user_id}")
                return user_id
        except:
            pass

    print("\nСОЗДАНИЕ НОВОГО ПОЛЬЗОВАТЕЛЯ")
    user_id = input("Введите ваше имя: ").strip()
    if not user_id:
        user_id = "player"
    print(f"Создан пользователь: {user_id}")
    return user_id


if __name__ == "__main__":
    user_id = select_user()

    with open("current_user.txt", "w") as f:
        f.write(user_id)

    from game.app import App
    App().run()