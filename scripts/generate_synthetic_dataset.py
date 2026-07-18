#!/usr/bin/env python3
"""Генерирует обязательный русскоязычный TTS corpus из versioned сценариев."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import edge_tts  # pyright: ignore[reportMissingImports]
from pydub import AudioSegment  # pyright: ignore[reportMissingImports]

Role = Literal["Оператор", "Клиент"]

_OPERATOR_VOICE = "ru-RU-SvetlanaNeural"
_CLIENT_VOICE = "ru-RU-DmitryNeural"
_TTS_RATE = "+0%"
_PAUSE_MS = 800
_TRAILING_SILENCE_MS = 1_000
_LICENSE = "LicenseRef-MTBank-Synthetic-EdgeTTS-Demo"
_ELIGIBLE_FOR = [
    "media_transport",
    "wer",
    "der",
    "role_accuracy",
    "speaker_attributed_wer",
]


@dataclass(frozen=True, slots=True)
class CallSpec:
    identifier: str
    filename: str
    sample_rate_hz: int
    utterances: tuple[tuple[Role, str], ...]

    @property
    def format(self) -> str:
        return Path(self.filename).suffix.lstrip(".")


CALLS = (
    CallSpec(
        identifier="synthetic-credit-consultation",
        filename="credit-consultation-16k.wav",
        sample_rate_hz=16_000,
        utterances=(
            ("Оператор", "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?"),
            ("Клиент", "Здравствуйте. Хочу подробно узнать про условия кредита наличными."),
            ("Оператор", "Конечно. Подскажите, пожалуйста, какая сумма вас интересует и на какой срок?"),
            ("Клиент", "Примерно десять тысяч рублей на один год. Деньги нужны для ремонта квартиры."),
            (
                "Оператор",
                "Поняла вас. Окончательная ставка и доступная сумма определяются после рассмотрения анкеты. Предварительное решение обычно занимает до пятнадцати минут.",
            ),
            ("Клиент", "Я уже клиент банка и получаю зарплату на карточку. Это влияет на условия?"),
            (
                "Оператор",
                "Для действующих клиентов могут быть доступны персональные условия. Точное предложение появится в приложении после заполнения заявки и проверки данных.",
            ),
            ("Клиент", "Какой примерно будет ежемесячный платёж?"),
            (
                "Оператор",
                "Предварительный платёж зависит от ставки и даты выдачи. Перед подписанием договора вы увидите полный график, общую стоимость кредита и сумму каждого платежа.",
            ),
            ("Клиент", "Можно ли погасить кредит раньше срока и есть ли за это штраф?"),
            (
                "Оператор",
                "Досрочное погашение доступно без штрафа. Перед операцией приложение покажет сумму основного долга и начисленных процентов на выбранную дату.",
            ),
            ("Клиент", "А страхование жизни обязательно?"),
            (
                "Оператор",
                "Страхование является добровольным. Отказ от него не запрещает подать заявку, однако условия предложения могут отличаться. Все варианты будут показаны до подтверждения.",
            ),
            ("Клиент", "Какие документы понадобятся?"),
            (
                "Оператор",
                "Обычно нужен документ, удостоверяющий личность, и сведения из анкеты. Банк может запросить дополнительные документы после рассмотрения заявки.",
            ),
            ("Клиент", "Я могу всё оформить дистанционно?"),
            (
                "Оператор",
                "Если предложение доступно в мобильном приложении, основные шаги можно пройти онлайн. В отдельных случаях потребуется посещение отделения.",
            ),
            ("Клиент", "Расскажите, пожалуйста, как найти заявку в приложении."),
            (
                "Оператор",
                "Откройте раздел продуктов, выберите кредиты, затем нажмите кнопку новой заявки. Проверьте контактные данные, сумму, срок и подтвердите согласие на обработку информации.",
            ),
            ("Клиент", "Если я начну заполнять анкету и закрою приложение, данные сохранятся?"),
            (
                "Оператор",
                "Черновик может сохраняться ограниченное время. Перед продолжением убедитесь, что сумма и срок не изменились, а также повторно прочитайте итоговые условия.",
            ),
            ("Клиент", "Можно ли получить консультацию после предварительного решения?"),
            (
                "Оператор",
                "Да. Вы можете снова позвонить в контакт-центр или обратиться в отделение. Сотрудник объяснит параметры предложения, но решение о подписании всегда остаётся за вами.",
            ),
            ("Клиент", "Хорошо. Тогда сегодня заполню заявку и внимательно посмотрю график платежей."),
            (
                "Оператор",
                "Отлично. Не сообщайте никому код подтверждения и данные для входа в приложение. Сотрудники банка не запрашивают пароль по телефону.",
            ),
            ("Клиент", "Понял, спасибо за предупреждение."),
            ("Оператор", "Есть ли у вас ещё вопросы по кредиту или приложению?"),
            ("Клиент", "Нет, теперь всё понятно."),
            ("Оператор", "Спасибо за обращение в МТБанк. Хорошего дня!"),
            ("Клиент", "Спасибо, до свидания."),
        ),
    ),
    CallSpec(
        identifier="synthetic-card-complaint-telephone",
        filename="card-complaint-8k.wav",
        sample_rate_hz=8_000,
        utterances=(
            ("Оператор", "Добрый день, служба поддержки МТБанка, меня зовут Анна."),
            ("Клиент", "Здравствуйте. Банкомат не выдал деньги, но сумма списалась с карточки."),
            (
                "Оператор",
                "Понимаю ваше беспокойство. Назовите только дату, примерное время и сумму операции. Полный номер карточки и коды подтверждения сообщать не нужно.",
            ),
            ("Клиент", "Сегодня около девяти утра я пытался снять двести рублей."),
            (
                "Оператор",
                "Спасибо. Иногда после неуспешной выдачи сумма автоматически возвращается. Я зарегистрирую обращение, чтобы специалисты проверили журнал банкомата.",
            ),
            ("Клиент", "Сколько времени займёт проверка?"),
            (
                "Оператор",
                "Срок зависит от результатов сверки. Статус обращения будет доступен в приложении, а уведомление поступит по вашему зарегистрированному каналу связи.",
            ),
            ("Клиент", "Карточку нужно блокировать?"),
            (
                "Оператор",
                "Если карточка остаётся у вас и других подозрительных операций нет, блокировка из-за одной ошибки банкомата обычно не требуется. При неизвестных списаниях заблокируйте её сразу в приложении.",
            ),
            ("Клиент", "Других списаний я не вижу."),
            (
                "Оператор",
                "Хорошо. Пожалуйста, сохраните чек, если банкомат его выдал, и не повторяйте операцию в этом устройстве до завершения проверки.",
            ),
            ("Клиент", "Понял. Я получу номер обращения?"),
            (
                "Оператор",
                "Да, номер появится в уведомлении. По нему можно уточнять статус, не передавая конфиденциальные данные.",
            ),
            ("Клиент", "Спасибо, буду ждать возврата."),
            ("Оператор", "Обращение зарегистрировано. Спасибо за звонок, до свидания."),
            ("Клиент", "До свидания."),
        ),
    ),
    CallSpec(
        identifier="synthetic-transfer-question",
        filename="transfer-question-16k.mp3",
        sample_rate_hz=16_000,
        utterances=(
            ("Оператор", "Добрый вечер, МТБанк, оператор Анна. Чем могу помочь?"),
            ("Клиент", "Здравствуйте. Перевод на другую карточку пока не дошёл получателю."),
            ("Оператор", "Уточните, пожалуйста, когда был сделан перевод и какой статус отображается в приложении."),
            ("Клиент", "Перевёл около часа назад. В истории написано, что операция выполнена."),
            (
                "Оператор",
                "Если операция завершена, срок зачисления может зависеть от банка получателя и платёжной системы. Иногда обновление баланса происходит не сразу.",
            ),
            ("Клиент", "Можно ли отменить перевод?"),
            (
                "Оператор",
                "Завершённый перевод обычно нельзя отменить автоматически. Если реквизиты указаны неверно, необходимо зарегистрировать обращение, но возврат не гарантируется и зависит от дальнейшей проверки.",
            ),
            ("Клиент", "Реквизиты правильные, получатель знакомый."),
            (
                "Оператор",
                "Тогда попросите получателя обновить приложение и проверить выписку, а не только текущий баланс. Если зачисления не будет до завтра, свяжитесь с нами повторно.",
            ),
            ("Клиент", "Комиссия была показана до подтверждения, это нормально?"),
            (
                "Оператор",
                "Да, размер комиссии должен отображаться до подтверждения операции. Итоговая сумма также сохраняется в деталях перевода.",
            ),
            ("Клиент", "Хорошо, подожду до завтра."),
            (
                "Оператор",
                "Не пересылайте никому код подтверждения и не повторяйте перевод по просьбе неизвестных лиц. Есть ещё вопросы?",
            ),
            ("Клиент", "Нет, спасибо за объяснение."),
            ("Оператор", "Спасибо за обращение. До свидания."),
            ("Клиент", "До свидания."),
        ),
    ),
    CallSpec(
        identifier="synthetic-mobile-app-security",
        filename="mobile-app-security-16k.ogg",
        sample_rate_hz=16_000,
        utterances=(
            ("Оператор", "Добрый день, контакт-центр МТБанка, меня зовут Анна."),
            ("Клиент", "Здравствуйте. Мне позвонили и попросили назвать код из сообщения для отмены кредита."),
            (
                "Оператор",
                "Не называйте код. Сотрудники банка не запрашивают одноразовые коды, пароль или данные для входа в приложение.",
            ),
            ("Клиент", "Я ничего не сообщил, но звонивший знал моё имя."),
            (
                "Оператор",
                "Правильно, что вы прекратили разговор. Знание имени не подтверждает, что звонит сотрудник банка. Заблокируйте неизвестный номер и не переходите по присланным ссылкам.",
            ),
            ("Клиент", "Нужно ли менять пароль?"),
            (
                "Оператор",
                "Если вы не вводили данные на стороннем сайте, риск ниже. Для дополнительной безопасности завершите неизвестные сессии и установите новый уникальный пароль.",
            ),
            ("Клиент", "Как проверить, нет ли заявки на кредит?"),
            (
                "Оператор",
                "Откройте приложение только через официальный значок, проверьте раздел заявок и уведомления. При неизвестной заявке немедленно свяжитесь с банком по номеру на обратной стороне карточки.",
            ),
            ("Клиент", "В приложении ничего подозрительного нет."),
            (
                "Оператор",
                "Хорошо. Я зафиксирую информацию о мошенническом звонке. Не устанавливайте программы удалённого доступа по просьбе звонящих.",
            ),
            ("Клиент", "Спасибо. Теперь буду внимательнее."),
            ("Оператор", "Есть ли другие вопросы по безопасности?"),
            ("Клиент", "Нет, всё понятно."),
            ("Оператор", "Спасибо за обращение. Берегите свои данные, до свидания."),
            ("Клиент", "До свидания."),
        ),
    ),
    CallSpec(
        identifier="synthetic-deposit-consultation",
        filename="deposit-consultation-16k.wav",
        sample_rate_hz=16_000,
        utterances=(
            ("Оператор", "Добрый день, МТБанк, оператор Анна. Слушаю вас."),
            ("Клиент", "Здравствуйте. Хочу разместить сбережения и сравниваю варианты вклада."),
            (
                "Оператор",
                "Подскажите, на какой срок вы готовы разместить деньги и важно ли иметь возможность частичного снятия?",
            ),
            ("Клиент", "Примерно на шесть месяцев, но часть суммы может понадобиться раньше."),
            (
                "Оператор",
                "Тогда стоит сравнить продукты с возможностью досрочного расходования и варианты с фиксированным сроком. Доходность и правила снятия у них отличаются.",
            ),
            ("Клиент", "Проценты выплачиваются каждый месяц?"),
            (
                "Оператор",
                "Порядок выплаты зависит от выбранного продукта. Возможна ежемесячная выплата или начисление в конце срока. Точные параметры отображаются в договоре.",
            ),
            ("Клиент", "Можно открыть вклад через приложение?"),
            (
                "Оператор",
                "Некоторые вклады доступны онлайн. Перед подтверждением приложение покажет ставку, срок, минимальную сумму, порядок выплаты и последствия досрочного закрытия.",
            ),
            ("Клиент", "Если закрыть раньше срока, проценты сохранятся?"),
            (
                "Оператор",
                "Это определяется условиями конкретного вклада. Проценты могут быть пересчитаны, поэтому обязательно прочитайте раздел о досрочном расторжении до открытия.",
            ),
            ("Клиент", "Понятно. Я сначала сравню варианты в приложении."),
            (
                "Оператор",
                "Хорошее решение. Если потребуется, мы поможем объяснить различия, но выбор продукта и подтверждение условий выполняете вы самостоятельно.",
            ),
            ("Клиент", "Спасибо за консультацию."),
            ("Оператор", "Пожалуйста. Есть ли у вас ещё вопросы?"),
            ("Клиент", "Нет, спасибо."),
            ("Оператор", "Спасибо за обращение в МТБанк. До свидания."),
            ("Клиент", "До свидания."),
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> tuple[float, int, int]:
    result = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return float(payload["format"]["duration"]), int(stream["sample_rate"]), int(stream["channels"])


async def _synthesize_utterance(text: str, role: Role, output: Path) -> None:
    voice = _OPERATOR_VOICE if role == "Оператор" else _CLIENT_VOICE
    await edge_tts.Communicate(text=text, voice=voice, rate=_TTS_RATE).save(str(output))


async def _generate_call(
    spec: CallSpec, *, audio_root: Path, reference_root: Path, temp_root: Path
) -> dict[str, object]:
    call_temp = temp_root / spec.identifier
    call_temp.mkdir(parents=True, exist_ok=True)
    combined = AudioSegment.silent(duration=250, frame_rate=spec.sample_rate_hz)
    segments: list[dict[str, object]] = []

    for index, (role, text) in enumerate(spec.utterances, start=1):
        utterance_path = call_temp / f"{index:03d}.mp3"
        await _synthesize_utterance(text, role, utterance_path)
        utterance = AudioSegment.from_file(utterance_path).set_channels(1).set_frame_rate(spec.sample_rate_hz)
        start = len(combined) / 1000
        combined += utterance
        end = len(combined) / 1000
        segments.append(
            {
                "id": f"{spec.identifier}-{index:03d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": role,
                "text": text,
            }
        )
        combined += AudioSegment.silent(duration=_PAUSE_MS, frame_rate=spec.sample_rate_hz)

    combined += AudioSegment.silent(duration=_TRAILING_SILENCE_MS, frame_rate=spec.sample_rate_hz)
    combined = combined.set_channels(1).set_frame_rate(spec.sample_rate_hz).set_sample_width(2)
    audio_path = audio_root / spec.filename
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    export_options: dict[str, object] = {"format": spec.format}
    if spec.format == "mp3":
        export_options["bitrate"] = "64k"
    elif spec.format == "ogg":
        export_options["codec"] = "libvorbis"
    combined.export(audio_path, **export_options)

    reference_path = reference_root / f"{spec.identifier}.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    duration, sample_rate, channels = _probe(audio_path)
    return {
        "id": spec.identifier,
        "kind": "speech_reference",
        "path": audio_path.relative_to(audio_root.parent).as_posix(),
        "sha256": _sha256(audio_path),
        "format": spec.format,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "duration_seconds": round(duration, 3),
        "license": _LICENSE,
        "provenance": (
            "Синтетический русский банковский диалог, написанный для задания и сгенерированный "
            f"Edge TTS {_OPERATOR_VOICE}/{_CLIENT_VOICE}, rate {_TTS_RATE}; реальных клиентов нет."
        ),
        "eligible_for": _ELIGIBLE_FOR,
        "excluded_from": [],
        "reference_path": reference_path.relative_to(audio_root.parent).as_posix(),
        "reference_sha256": _sha256(reference_path),
        "speaker_count": 2,
    }


async def generate(root: Path, *, keep_transport: bool) -> None:
    audio_root = root / "synthetic"
    reference_root = root / "references"
    temp_root = root.parent / "tmp" / "synthetic-dataset"
    if audio_root.exists():
        shutil.rmtree(audio_root)
    if reference_root.exists():
        shutil.rmtree(reference_root)
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)

    generated = []
    for spec in CALLS:
        generated.append(
            await _generate_call(spec, audio_root=audio_root, reference_root=reference_root, temp_root=temp_root)
        )

    transport: list[object] = []
    manifest_path = root / "manifest.yaml"
    if keep_transport and manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        transport = [entry for entry in current.get("entries", []) if entry.get("kind") == "transport_only"]
    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "mtbank-speech-evaluation",
            "status": "release_ready",
            "source": "authored synthetic Edge TTS corpus permitted by the assignment README",
            "normalization": {"sample_rate_hz": 16000, "channels": 1, "codec": "pcm_s16le"},
        },
        "entries": [*generated, *transport],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(temp_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("test_data"))
    parser.add_argument("--drop-transport", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(generate(arguments.root.resolve(), keep_transport=not arguments.drop_transport))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
