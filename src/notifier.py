"""
消息推送引擎 (Notifier)
===================================================
专为量化交易研报与调仓信号设计的手机端触达引擎。
集成多通道分发协议：
1. PushPlus (推送加) - 免费 200 条/日，微信服务号直达，个人开发者首选
2. Server酱 (Turbo 版) - 极简微信推送通道
3. 企业微信群机器人 (WeCom Webhook) - 机构与团队级零限流通道
4. 钉钉/飞书群机器人 (DingTalk/Feishu) - 办公场景高可读性 Markdown 渲染
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_LOG_PATH = LOGS_DIR / "model_run.log"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


def setup_logger(log_file: Path = DEFAULT_LOG_PATH, name: str = "Notifier") -> logging.Logger:
    """配置并获取统一日志记录器"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Notifier:
    """
    量化信号与风控报告多通道消息推送引擎
    """

    def __init__(self):
        self.logger = setup_logger(name="Notifier")
        # 读取推送配置环境变量
        self.pushplus_token = os.getenv("PUSHPLUS_TOKEN") or os.getenv("WECHAT_PUSH_TOKEN")
        self.serverchan_key = os.getenv("SERVERCHAN_KEY") or os.getenv("SCT_KEY")
        self.wecom_webhook = os.getenv("WECOM_WEBHOOK") or os.getenv("WECHAT_WEBHOOK_URL")
        self.dingtalk_webhook = os.getenv("DINGTALK_WEBHOOK")
        self.feishu_webhook = os.getenv("FEISHU_WEBHOOK")

    def send_pushplus(self, title: str, content: str) -> bool:
        """
        通过 PushPlus (推送加) 发送微信通知

        :param title: 消息标题
        :param content: Markdown 格式正文
        :return: 是否发送成功
        """
        if not self.pushplus_token or "your_" in self.pushplus_token:
            return False

        url = "http://www.pushplus.plus/send"
        payload = {
            "token": self.pushplus_token,
            "title": title,
            "content": content,
            "template": "markdown",
            "channel": "wechat",
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            result = resp.json()
            if result.get("code") == 200:
                self.logger.info("【PushPlus】微信消息推送成功！")
                return True
            else:
                self.logger.warning(f"【PushPlus】推送返回异常: {result.get('msg')}")
                return False
        except Exception as e:
            self.logger.warning(f"【PushPlus】网络请求失败: {e}")
            return False

    def send_serverchan(self, title: str, content: str) -> bool:
        """
        通过 Server酱 Turbo 版发送微信通知

        :param title: 消息标题
        :param content: Markdown 格式正文
        :return: 是否发送成功
        """
        if not self.serverchan_key or "your_" in self.serverchan_key:
            return False

        url = f"https://sctapi.ftqq.com/{self.serverchan_key}.send"
        payload = {"title": title, "desp": content}
        try:
            resp = requests.post(url, data=payload, timeout=10)
            result = resp.json()
            if result.get("code") == 0:
                self.logger.info("【Server酱】微信消息推送成功！")
                return True
            else:
                self.logger.warning(f"【Server酱】推送返回异常: {result.get('message')}")
                return False
        except Exception as e:
            self.logger.warning(f"【Server酱】网络请求失败: {e}")
            return False

    def send_wecom(self, title: str, content: str) -> bool:
        """
        通过企业微信群机器人 Webhook 发送通知

        :param title: 标题
        :param content: Markdown 内容
        :return: 是否发送成功
        """
        if not self.wecom_webhook or "http" not in self.wecom_webhook:
            return False

        payload = {
            "msgtype": "markdown",
            "markdown": {"content": f"### {title}\n\n{content}"},
        }
        try:
            resp = requests.post(self.wecom_webhook, json=payload, timeout=10)
            result = resp.json()
            if result.get("errcode") == 0:
                self.logger.info("【企业微信】群机器人消息推送成功！")
                return True
            else:
                self.logger.warning(f"【企业微信】推送返回异常: {result}")
                return False
        except Exception as e:
            self.logger.warning(f"【企业微信】推送请求失败: {e}")
            return False

    def send_dingtalk(self, title: str, content: str) -> bool:
        """
        通过钉钉群机器人 Webhook 发送通知

        :param title: 标题
        :param content: Markdown 正文
        :return: 是否发送成功
        """
        if not self.dingtalk_webhook or "http" not in self.dingtalk_webhook:
            return False

        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": f"### {title}\n\n{content}"},
        }
        try:
            resp = requests.post(self.dingtalk_webhook, json=payload, timeout=10)
            result = resp.json()
            if result.get("errcode") == 0:
                self.logger.info("【钉钉】群机器人消息推送成功！")
                return True
            else:
                self.logger.warning(f"【钉钉】推送异常: {result}")
                return False
        except Exception as e:
            self.logger.warning(f"【钉钉】网络请求失败: {e}")
            return False

    def send_feishu(self, title: str, content: str) -> bool:
        """
        通过飞书群机器人 Webhook 发送通知

        :param title: 标题
        :param content: 内容
        :return: 是否发送成功
        """
        if not self.feishu_webhook or "http" not in self.feishu_webhook:
            return False

        payload = {
            "msg_type": "text",
            "content": {"text": f"【{title}】\n\n{content}"},
        }
        try:
            resp = requests.post(self.feishu_webhook, json=payload, timeout=10)
            result = resp.json()
            if result.get("code") == 0 or result.get("StatusCode") == 0:
                self.logger.info("【飞书】群机器人消息推送成功！")
                return True
            else:
                self.logger.warning(f"【飞书】推送异常: {result}")
                return False
        except Exception as e:
            self.logger.warning(f"【飞书】网络请求失败: {e}")
            return False

    def broadcast(self, title: str, content: str) -> bool:
        """
        多通道广播分发：
        自动检测已配置的通道进行并行或顺序推送；若未配置任何通道，则打印至终端与日志

        :param title: 研报标题
        :param content: Markdown 报告正文
        :return: 至少一个通道发送成功返回 True
        """
        self.logger.info(f"正在广播量化推送消息: 《{title}》...")
        sent_any = False

        # 1. 尝试 PushPlus
        if self.send_pushplus(title, content):
            sent_any = True

        # 2. 尝试 Server酱
        if self.send_serverchan(title, content):
            sent_any = True

        # 3. 尝试企业微信
        if self.send_wecom(title, content):
            sent_any = True

        # 4. 尝试钉钉
        if self.send_dingtalk(title, content):
            sent_any = True

        # 5. 尝试飞书
        if self.send_feishu(title, content):
            sent_any = True

        if not sent_any:
            self.logger.warning(
                "未检测到有效的推送 Token (未配置 .env 文件或 Token 为空)。\n"
                "以下为本次生成的完整推送报告预览："
            )
            print("\n" + "=" * 50)
            print(f"【{title}】\n")
            print(content)
            print("=" * 50 + "\n")

        return sent_any


if __name__ == "__main__":
    notifier = Notifier()
    test_title = "【AShare Quant】测试消息推送"
    test_content = (
        "### 🟢 量化交易系统测试推送\n\n"
        "- **触发时间**: 2026-11-01 15:30:00\n"
        "- **系统状态**: 正常运行中\n"
        "- **推送通道**: 自动检测\n\n"
        "> 本系统采用 AI 机器学习算法驱动，每日盘后提供中证 500 优选信号。"
    )
    notifier.broadcast(test_title, test_content)
