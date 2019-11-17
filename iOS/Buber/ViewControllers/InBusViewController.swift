//
//  InBusViewController.swift
//  Buber
//
//  Created by Alexey Salangin on 11/17/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit

protocol InBusViewControllerDelegate: AnyObject {
    func inBusViewControllerDidTapFinishButton(_ vc: InBusViewController)
}

final class InBusViewController: UIViewController {
    weak var delegate: InBusViewControllerDelegate?

    private let finishButton: UIButton
    private let starsImageView = UIImageView()
    private let rateUsLabel = UILabel()

    init() {
        self.finishButton = Self.makeFinishButton()
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        self.view.backgroundColor = .white
        self.setupSubviews()
    }

    private func setupSubviews() {
        let stackView = UIStackView()
        stackView.axis = .vertical

        self.view.addSubview(stackView)
        stackView.snp.makeConstraints { make in
            make.left.equalToSuperview().offset(16)
            make.right.equalToSuperview().offset(-16)
            make.bottom.equalToSuperview().offset(-46)
        }

        stackView.addArrangedSubview(self.rateUsLabel)
        stackView.addArrangedSubview(self.starsImageView)
        stackView.addArrangedSubview(self.finishButton)

        self.rateUsLabel.text = "How was your experiense with us?"
        self.rateUsLabel.font = UIFont.appMedium(24)
        self.rateUsLabel.textAlignment = .center
        self.rateUsLabel.numberOfLines = 0

        self.starsImageView.image = UIImage(named: "stars")
        self.starsImageView.snp.makeConstraints { make in
            make.height.equalTo(47)
        }
        self.starsImageView.contentMode = .scaleAspectFit

        stackView.setCustomSpacing(24, after: self.rateUsLabel)
        stackView.setCustomSpacing(110, after: self.starsImageView)

        self.finishButton.addTarget(self, action: #selector(self.didTapFinishButton), for: .touchUpInside)
    }

    @objc
    private func didTapFinishButton() {
        self.delegate?.inBusViewControllerDidTapFinishButton(self)
    }

    private static func makeFinishButton() -> UIButton {
        let button = UIButton()
        button.setBackgroundImage(UIImage.imageWith(color: .cancel), for: .normal)
        button.layer.cornerRadius = 10
        button.layer.masksToBounds = true
        let attributedTitle = NSAttributedString(
            string: "Finish the route",
            attributes: [
                .foregroundColor: UIColor.white,
                .font: UIFont.appMedium(16)
            ]
        )
        button.setAttributedTitle(attributedTitle, for: .normal)
        button.snp.makeConstraints { make in
            make.height.equalTo(46)
        }
        return button
    }
}
