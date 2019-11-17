//
//  PaymentView.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit

final class PaymentView: UIView {
    private let bankCardStackView = UIStackView()
    private let changeButton: UIButton
    private let paymentSystemLogoImageView = UIImageView()
    private let bankCardLabel = UILabel()

    init() {
        self.changeButton = Self.makeChangeButton()

        super.init(frame: .zero)

        self.setupSubviews()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupSubviews() {
        self.addSubview(self.bankCardStackView)
        self.addSubview(self.changeButton)

        self.bankCardStackView.snp.makeConstraints { make in
            make.left.equalToSuperview().offset(8)
            make.centerY.equalToSuperview()
            make.height.equalTo(19)
        }

        self.changeButton.snp.makeConstraints { make in
            make.left.equalTo(self.bankCardStackView.snp.right).offset(20)
            make.top.equalToSuperview()
            make.right.equalToSuperview()
            make.bottom.equalToSuperview()
            make.height.equalTo(28)
            make.width.equalTo(105)
        }
            
        self.setupBankCardStackView()
    }

    private func setupBankCardStackView() {
        self.bankCardStackView.addArrangedSubview(self.paymentSystemLogoImageView)
        self.bankCardStackView.addArrangedSubview(self.bankCardLabel)
        self.bankCardStackView.axis = .horizontal
        self.bankCardStackView.spacing = 12

        self.paymentSystemLogoImageView.image = UIImage(named: "mastercard")
        self.paymentSystemLogoImageView.snp.makeConstraints { make in
            make.width.equalTo(97)
        }
        self.bankCardLabel.text = "2384"
        self.bankCardLabel.font = UIFont.appRegular(16)
    }

    private static func makeChangeButton() -> UIButton {
        let button = UIButton()
        let image = UIImage.imageWith(color: .changeButton)
        button.setBackgroundImage(image, for: .normal)
        let font = UIFont.appRegular(12)
        let attributedTitle = NSAttributedString(string: "Change", attributes: [.foregroundColor: UIColor.blackText, .font: font])
        button.setAttributedTitle(attributedTitle, for: .normal)
        button.layer.masksToBounds = true
        button.layer.cornerRadius = 4
        return button
    }
}
